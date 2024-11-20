import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="egamma",                     # This is the name of the package
    version="1.0.2",
    author="Frode Drevland",                     # Full name of the author
    author_email='frode.drevland@ntnu.no',
    description="Implementation of the expanded gamma distribution",
    url='https://github.com/FrodeDrevland/egamma',
    long_description=long_description,      # Long description read from the readme file
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    install_requires=['scipy', 'numpy']                     # Install other dependencies if any
)