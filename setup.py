from setuptools import setup, find_packages

with open("requirements.txt")as f:
    requirements = f.read().splitlines()


setup(

    name="firstMLOpsProject",
    version= 0.1,
    author="tushar kale",
    author_email="tusharkale816@gmail.com",
    packages=find_packages(),
    install_requires = requirements,
    

)
