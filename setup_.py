from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()


setup(name='binpan',
      version='1.0.1',
      url='https://github.com/nand0san/binpan_studio',
      license='MIT',
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.9",
      ],
      author='Fernando Alfonso',
      author_email='hancaidolosdos@hotmail.com',
      description='Binance API wrapper with backtesting tools.',
      long_description_content_type="text/markdown",
      package_dir={"": "."},
      packages=find_packages(where="./handlers"),
      )
