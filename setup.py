import setuptools

f = open("README.md","r", encoding = 'utf8')
long_description = f.read()

setuptools.setup(
    name="hsipl_tools",
    version="0.0.3",
    description="Tools for Hyperspectral Signal and Image Processing Laboratory of YunTech, Taiwan",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yuchi",
    author_email="jackey860517@gmail.com",   
    url="https://github.com/ek2061/hsipl_tools",
    packages=['hsipl_tools'],
    install_requires = [
        "numpy >= 1.14.5",
        "matplotlib",
        "scipy",
        "pandas",
        "scikit-learn"
    ],  
    python_requires='>=3.6',
)