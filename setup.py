# coding: utf-8
# create by tongshiwei on 2019/6/25

from setuptools import setup, find_packages

test_deps = [
    'pytest>=4',
    'pytest-cov>=2.6.0',
    'pytest-flake8',
]

setup(
    name='EduSim',
    version='0.1.0',
    extras_require={
        'test': test_deps,
    },
    packages=find_packages(),
    python_requires='>=3.6',
    long_description='Refer to full documentation https://github.com/bigdata-ustc/EduSim/blob/master/README.md'
                     ' for detailed information.',
    description='This project aims to '
                'provide some offline simulators for training and testing recommender systems of education.',
    install_requires=[
        'gym',
        'longling>=1.3.19',
        'tqdm',
        'networkx',
        'pandas',
        'scikit-learn',
        'numpy',
        'tensorboardX',
        'tensorboard',
    ]  # And any other dependencies foo needs
)
