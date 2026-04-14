from setuptools import find_packages, setup

setup(
    name="智能放疗计划系统",
    version="0.1.0",
    description="A Python project with AI script, notebooks, and MATLAB helpers",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
)
