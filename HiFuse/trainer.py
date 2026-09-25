import torch
import torch.nn.functional as F
from tqdm import tqdm

from .model import HiFuseNetwork
from .preprocess import adjacent_matrix_preprocessing


PLATFORM_PRESETS = {
    "SPOTS": (600, [1, 5, 1, 1, 0.5, 0.25, 0.5]),
    "Stereo-CITE-seq": (1500, [1, 10, 1, 10, 0.25, 1.0, 1.0]),
    "10x": (200, [1, 5, 1, 10, 0.2, 0.1, 0.1]),
    "Spatial-epigenome-transcriptome": (1600, [1, 5, 1, 1, 0.1, 0.05, 0.1]),
}


class HiFuseTrainer:
    def __init__(
        self,
        data,
        datatype="SPOTS",
        device=torch.device("cpu"),
        random_seed=2022,
        learning_rate=0.0001,
        weight_decay=0.0,
        epochs=None,
        dim_input=3000,
        dim_output=64,
        weight_factors=None,
        loss_corr_weight=None,
        loss_consensus_weight=None,
        loss_fusion_weight=None,
        loss_contrastive_weight=None,
        pretrain_ratio=0.0,
    ):
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors
        self.loss_corr_weight = loss_corr_weight
        self.loss_consensus_weight = loss_consensus_weight
        self.loss_fusion_weight = loss_fusion_weight
        self.loss_contrastive_weight = loss_contrastive_weight
        self.temperature = 1.0
        self.pretrain_ratio = pretrain_ratio

        self.adata_omics1 = self.data["adata_omics1"]
        self.adata_omics2 = self.data["adata_omics2"]
        self.adj = adjacent_matrix_preprocessing(
            self.adata_omics1,
            self.adata_omics2,
        )
        self.adj_spatial_omics1 = self.adj["adj_spatial_omics1"].to(self.device)
        self.adj_spatial_omics2 = self.adj["adj_spatial_omics2"].to(self.device)
        self.adj_feature_omics1 = self.adj["adj_feature_omics1"].to(self.device)
        self.adj_feature_omics2 = self.adj["adj_feature_omics2"].to(self.device)

        self.features_omics1 = torch.FloatTensor(
            self.adata_omics1.obsm["feat"].copy()
        ).to(self.device)
        self.features_omics2 = torch.FloatTensor(
            self.adata_omics2.obsm["feat"].copy()
        ).to(self.device)
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output

        preset_epochs, preset_weights = PLATFORM_PRESETS.get(
            self.datatype,
            (600, [1, 5, 1, 1, 1.0, 1.0, 1.0]),
        )
        if self.epochs is None:
            self.epochs = preset_epochs
        if self.weight_factors is None:
            self.weight_factors = preset_weights[:4]
        else:
            self.weight_factors = list(self.weight_factors)
        if self.loss_corr_weight is not None:
            self.weight_factors[2] = self.loss_corr_weight
            self.weight_factors[3] = self.loss_corr_weight
        if self.loss_consensus_weight is None:
            self.loss_consensus_weight = preset_weights[4]
        if self.loss_fusion_weight is None:
            self.loss_fusion_weight = preset_weights[5]
        if self.loss_contrastive_weight is None:
            self.loss_contrastive_weight = preset_weights[6]

        self.model = None
        self.optimizer = None

    @staticmethod
    def _normalize(embedding):
        return F.normalize(embedding, p=2, eps=1e-12, dim=1)

    def _compute_modality_weights(self, normalized_views, normalized_fusion):
        n_samples = normalized_fusion.shape[0]
        if n_samples <= 0:
            return torch.full(
                (len(normalized_views),),
                1.0 / len(normalized_views),
                device=self.device,
            )

        fusion_similarity = torch.matmul(normalized_fusion, normalized_fusion.T)
        weights = []
        for view in normalized_views:
            view_similarity = torch.matmul(view, view.T)
            cross_similarity = torch.matmul(view, normalized_fusion.T)
            discrepancy = (
                torch.sum(view_similarity)
                + torch.sum(fusion_similarity)
                - 2.0 * torch.sum(cross_similarity)
            ) / float(n_samples * n_samples)
            weights.append(torch.exp(-discrepancy))

        weights = torch.stack(weights)
        return weights / (torch.sum(weights) + 1e-12)

    def _contrastive_alignment_loss(self, normalized_fusion, normalized_view):
        n_samples = normalized_fusion.shape[0]
        if n_samples <= 1:
            return torch.tensor(0.0, device=self.device)

        logits = torch.matmul(normalized_fusion, normalized_view.T)
        logits = logits / max(self.temperature, 1e-6)
        positives = torch.diag(logits)
        identity = torch.eye(n_samples, device=logits.device, dtype=torch.bool)
        negatives = logits.masked_fill(identity, float("-inf"))
        return torch.mean(-positives + torch.logsumexp(negatives, dim=1))

    def _build_model(self):
        self.model = HiFuseNetwork(
            self.dim_input1,
            self.dim_output1,
            self.dim_input2,
            self.dim_output2,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()),
            self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _ensure_model(self):
        if self.model is None or self.optimizer is None:
            self._build_model()

    def get_model_stats(self):
        self._ensure_model()
        total_parameters = 0
        trainable_parameters = 0
        parameter_bytes = 0
        buffer_bytes = 0

        for parameter in self.model.parameters():
            count = int(parameter.numel())
            total_parameters += count
            if parameter.requires_grad:
                trainable_parameters += count
            parameter_bytes += count * int(parameter.element_size())
        for buffer in self.model.buffers():
            buffer_bytes += int(buffer.numel()) * int(buffer.element_size())

        return {
            "model_total_parameters": int(total_parameters),
            "model_trainable_parameters": int(trainable_parameters),
            "model_parameter_bytes": int(parameter_bytes),
            "model_buffer_bytes": int(buffer_bytes),
            "model_size_bytes": int(parameter_bytes + buffer_bytes),
        }

    def _forward_model(self):
        return self.model(
            self.features_omics1,
            self.features_omics2,
            self.adj_spatial_omics1,
            self.adj_feature_omics1,
            self.adj_spatial_omics2,
            self.adj_feature_omics2,
        )

    def _base_losses(self, outputs):
        self.loss_recon_omics1 = F.mse_loss(
            self.features_omics1,
            outputs["reconstruction1"],
        )
        self.loss_recon_omics2 = F.mse_loss(
            self.features_omics2,
            outputs["reconstruction2"],
        )
        self.loss_corr_omics1 = F.mse_loss(
            outputs["modality1"],
            outputs["cycle1"],
        )
        self.loss_corr_omics2 = F.mse_loss(
            outputs["modality2"],
            outputs["cycle2"],
        )
        return (
            self.weight_factors[0] * self.loss_recon_omics1
            + self.weight_factors[1] * self.loss_recon_omics2
            + self.weight_factors[2] * self.loss_corr_omics1
            + self.weight_factors[3] * self.loss_corr_omics2
        )

    def _refinement_losses(self, outputs):
        consensus1 = self._normalize(outputs["consensus1"])
        consensus2 = self._normalize(outputs["consensus2"])
        attention_fusion = self._normalize(outputs["attention_fusion"])
        complementary = self._normalize(outputs["complementary_fusion"])

        modality_weights = self._compute_modality_weights(
            [consensus1, consensus2],
            complementary,
        )
        self.loss_consensus = F.mse_loss(consensus1, consensus2)
        self.loss_fusion = F.mse_loss(attention_fusion, complementary)
        contrastive1 = self._contrastive_alignment_loss(
            complementary,
            consensus1,
        )
        contrastive2 = self._contrastive_alignment_loss(
            complementary,
            consensus2,
        )
        self.loss_contrastive = (
            modality_weights[0] * contrastive1 + modality_weights[1] * contrastive2
        )
        return modality_weights

    def train(self):
        self._ensure_model()
        self.model.train()
        pretrain_epochs = int(self.epochs * self.pretrain_ratio)
        self.training_history = []

        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            outputs = self._forward_model()
            loss = self._base_losses(outputs)

            self.loss_consensus = torch.tensor(0.0, device=self.device)
            self.loss_fusion = torch.tensor(0.0, device=self.device)
            self.loss_contrastive = torch.tensor(0.0, device=self.device)
            modality_weights = torch.tensor([0.5, 0.5], device=self.device)

            if epoch >= pretrain_epochs:
                modality_weights = self._refinement_losses(outputs)
                loss = loss + (
                    self.loss_consensus_weight * self.loss_consensus
                    + self.loss_fusion_weight * self.loss_fusion
                    + self.loss_contrastive_weight * self.loss_contrastive
                )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.training_history.append(
                {
                    "epoch": int(epoch + 1),
                    "loss_total": float(loss.item()),
                    "loss_recon_omics1": float(self.loss_recon_omics1.item()),
                    "loss_recon_omics2": float(self.loss_recon_omics2.item()),
                    "loss_corr_omics1": float(self.loss_corr_omics1.item()),
                    "loss_corr_omics2": float(self.loss_corr_omics2.item()),
                    "loss_consensus": float(self.loss_consensus.item()),
                    "loss_fusion": float(self.loss_fusion.item()),
                    "loss_contrastive": float(self.loss_contrastive.item()),
                    "view_weight_omics1": float(modality_weights[0].item()),
                    "view_weight_omics2": float(modality_weights[1].item()),
                }
            )

        print("HiFuse training finished!\n")
        with torch.no_grad():
            self.model.eval()
            outputs = self._forward_model()

        modality1 = self._normalize(outputs["modality1"])
        modality2 = self._normalize(outputs["modality2"])
        joint = self._normalize(outputs["joint"])

        modality_reliability = self._compute_modality_weights(
            [
                self._normalize(outputs["consensus1"]),
                self._normalize(outputs["consensus2"]),
            ],
            self._normalize(outputs["complementary_fusion"]),
        )

        return {
            "HiFuse": joint.detach().cpu().numpy(),
            "emb_latent_omics1": modality1.detach().cpu().numpy(),
            "emb_latent_omics2": modality2.detach().cpu().numpy(),
            "alpha_omics1": outputs["graph_weights1"].detach().cpu().numpy(),
            "alpha_omics2": outputs["graph_weights2"].detach().cpu().numpy(),
            "alpha": outputs["modality_weights"].detach().cpu().numpy(),
            "alpha_global": outputs["path_weights"].detach().cpu().numpy(),
            "modality_reliability": modality_reliability.detach().cpu().numpy(),
            "training_history": self.training_history,
        }
