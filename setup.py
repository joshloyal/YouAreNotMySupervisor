from setuptools import setup


PACKAGES = [
        'mysuper',
        'mysuper.datasets',
        'mysuper.tests',
]

def setup_package():
    setup(
        name="YouAreNotMySupervisor",
        version='0.1.0',
        description='Clustering Sandbox in Python',
        author='Joshua D. Loyal',
        url='https://github.com/joshloyal/YouAreNotMySupervisor',
        license='MIT',
        install_requires=['numpy', 'scipy', 'scikit-learn', 'hdbscan'],
        packages=PACKAGES,
    )


if __name__ == '__main__':
    setup_package()
