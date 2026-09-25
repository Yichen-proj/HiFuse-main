from setuptools import find_packages, setup


setup(
    name="HiFuse",
    version="0.1.0",
    description="Hierarchical fusion for spatial multi-omics integration",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "anndata>=0.8.0",
        "matplotlib>=3.4.2",
        "numpy>=1.22.3",
        "pandas>=1.4.2",
        "scanpy>=1.9.1",
        "scikit-learn>=1.1.1",
        "scipy>=1.8.1",
        "seaborn>=0.11.0",
        "torch>=1.8.0",
        "tqdm>=4.64.0",
    ],
    include_package_data=True,
    zip_safe=False,
)
