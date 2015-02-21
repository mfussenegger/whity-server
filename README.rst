

Development Setup
=================


Create a virtualenv::

    python2 -m virtualenv .venv

Install numpy in the virtualenv::

    .venv/bin/pip install numpy


Use the virtualenv python to bootstrap buildout::

    .venv/bin/python bootstrap.py

Run buildout::

    bin/buildout -N
