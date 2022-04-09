import os
from glob import glob
from setuptools import setup, find_packages

dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(dir, 'README.md')) as f:
    long_description = f.read()

setup(name='resize',
      version='0.1.3',
      author='Shuo Han',
      description='Resize an image with correct sampling coordinates.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author_email='shuo_han@outlook.com',
      url='https://github.com/shuohan/resize',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy'],
      extras_require={
            'scipy': 'scipy',
            'pytorch': 'torch>=1.10.0'
      },
      python_requires='>=3.7',
      include_package_data=True,
      classifiers=['Programming Language :: Python :: 3',
                   'License :: OSI Approved :: MIT License',
                   'Operating System :: OS Independent'])
