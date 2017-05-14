"""Setup file for skrvm."""
import os
import sys

from setuptools import setup, find_packages

import skrvm

version = skrvm.__version__

if sys.argv[-1] == 'publish':
    if os.system("pip freeze | grep wheel"):
        print("wheel not installed.\nUse `pip install wheel`.\nExiting.")
        sys.exit()
    if os.system("pip freeze | grep twine"):
        print("twine not installed.\nUse `pip install twine`.\nExiting.")
        sys.exit()
    os.system("python setup.py sdist bdist_wheel")
    os.system("twine upload dist/*")
    print("You probably want to also tag the version now:")
    print("  git tag -a %s -m 'version %s'" % (version, version))
    print("  git push --tags")
    sys.exit()

setup(
    name='scikit-rvm',
    version=version,
    description=(
        'Relevance Vector Machine implementation using the scikit-learn API'
    ),
    url='https://github.com/JamesRitchie/scikit-rvm',
    author='James Ritchie',
    author_email='james.a.ritchie@gmail.com',
    license='BSD',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        'numpy>=1.9.2',
        'scipy>=0.15.1',
        'scikit-learn>=0.16.1'
    ],
    test_suite='tests',
    tests_require=[
        'coverage>=3.7.1'
    ],
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.4',
    ]
)
