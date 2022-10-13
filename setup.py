from setuptools import setup
import pathlib
import configparser

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

readme = open('README.md', 'r')
README_TEXT = readme.read()
readme.close()

read_required = open('requirements.txt', 'r')
REQUIRED = read_required.read()
read_required.close()

my_version = "0.2.26"

setup(name='binpan',
      version=my_version,
      url='https://github.com/nand0san/binpan_studio',
      license='MIT',
      install_requires=REQUIRED,
      python_requires='>=3.7.9',
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.7",
      ],
      author='Fernando Alfonso',
      author_email='hancaidolosdos@hotmail.com',
      description='Binance API wrapper with backtesting tools.',
      long_description=README_TEXT,
      long_description_content_type="text/markdown",
      package_dir={"": "."},
      packages=["handlers", "binpan"]
      )
