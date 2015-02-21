#!/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import setup

requirements = [
    'tornado',
    'jinja2'
]

setup(
    name='whity',
    packages=['whity_server','whity_client'],
    package_dir={
        '': 'src'
    },
    entry_points={
        'console_scripts': [
            'whity_server = whity_server.main:main',
            'whity_client = whity_client.main:main',
            'convert = whity_server.main:convert',
        ],
    },
    install_requires=requirements
)
