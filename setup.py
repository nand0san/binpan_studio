from setuptools import setup
import pathlib

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
#
# req = os.path.abspath()
#
# with open(req) as f:
#     required = f.read().splitlines()

"""
      install_requires=[
            'requests~=2.28.0',
            'pytz~=2022.1',
            'py-cpuinfo~=8.0.0',
            'numpy==1.23.0',
            'pandas~=1.4.2',
            'plotly~=5.9.0',
            'tqdm~=4.64.0',
            'pycryptodome==3.15.0',
            'pandas-ta==0.3.14b0'
      ],
"""

setup(name='binpan',
      version='0.0.13',
      url='https://github.com/nand0san/binpan_studio',
      license='MIT',
      install_requires=REQUIRED,
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.9",
      ],
      author='Fernando Alfonso',
      author_email='hancaidolosdos@hotmail.com',
      description='Binance API wrapper with backtesting tools.',
      long_description=README_TEXT,
      long_description_content_type="text/markdown",
      package_dir={"": "."},
      packages=[
            ".",
            "handlers"
      ]
      )
