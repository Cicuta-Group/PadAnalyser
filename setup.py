from setuptools import find_namespace_packages, setup, find_packages

# required for us to install in ediable mode

if __name__ == '__main__':
    setup(
        packages=find_namespace_packages(where='src'),
        package_dir={"": "src"},
    )
