import setuptools

# required for us to install in ediable mode

if __name__ == '__main__':

    # install dependencies from requirements.txt file
    requirements = []
    with open('requirements.txt', 'r') as fh:
        for line in fh:
            requirements.append(line.strip())
    
    setuptools.setup(
        packages=setuptools.find_namespace_packages(where='src'),
        package_dir={"": "src"},
        install_requires=requirements,
        setup_requires=requirements,
    )
