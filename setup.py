#!/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import setup


requirements = [
    'tornado',
    'jinja2'
]


setup(
    name='whity_server',
    packages=['whity_server'],
    package_dir={
        '': 'src'
    },
    entry_points={
        'console_scripts': [
            'whity_server = whity_server.main:main',
        ],
    },
    install_requires=requirements
)
