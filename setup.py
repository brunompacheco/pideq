import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pideq",
    version="0.1.0",
    author="Bruno M. Pacheco",
    author_email="mpacheco.bruno@gmail.com",
    description="Physics-Informed Deep Equilibrium Models. With an application to the 4 tanks system.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brunompacheco/pideq",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)