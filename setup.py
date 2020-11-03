import setuptools

f = open("README.md","r", encoding = 'utf8')
long_description = f.read()

setuptools.setup(
    name="hsipl_tools",
    version="0.0.1",
    author="Yuchi",
    author_email="jackey860517@gmail.com",
    description="Tools for Hyperspectral Signal and Image Processing Laboratory of YunTech, Taiwan",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ek2061/hsipl_tools",
    packages=[''],
    requires = [
        "numpy",
        "matplotlib",
        "scipy",
        "pandas"
    ],
    license='MIT',
    python_requires='>=3.6',
)