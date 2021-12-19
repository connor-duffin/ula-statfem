import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sfmcmc",
    version="0.0.1",
    author="Connor Duffin",
    author_email="connor.p.duffin@gmail.com",
    description="Statistical Finite elements via Markov Chain Monte Carlo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/connor-duffin/sfmcmc",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy", "scipy", "fenics", "scikit-sparse", "petsc4py", "pyamg",
        "pytest"
    ])
