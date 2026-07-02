import os
from setuptools import find_packages, setup

# Version variable - will be updated during build
VERSION = "0.4.0"

def read_requirements(filename):
    """Read requirements from a file."""
    requirements = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Expand environment variables
                    line = os.path.expandvars(line)
                    requirements.append(line)
    return requirements

if __name__ == '__main__':
    # Read requirements
    reqs = read_requirements('requirements.txt')

    # Read README for long description
    long_description = ""
    if os.path.exists("../README.md"):
        with open("../README.md", "r") as fh:
            long_description = fh.read()

    setup(
        name='agent-builder',
        version=VERSION,
        packages=find_packages(include=['agent_builder', 'agent_builder.*']),
        package_data={
            'agent_builder': ['*.py', '**/*.py', '**/*.yaml', '**/*.json', '**/*.txt'],
        },
        install_requires=reqs,
        url="https://prod-gitlab.sprinklr.com/sprinklr/machine-learning/agent_builder",
        license='MIT',
        author='KKR',
        author_email='kapil.sharma@sprinklr.com, raghav.garg@sprinklr.com, karan.gupta1@sprinklr.com',
        description="Agent Orchestration Made Easy - Build, compose, and invoke custom AI Agents",
        long_description=long_description,
        long_description_content_type='text/markdown',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Operating System :: OS Independent',
        ],
        include_package_data=True,
        python_requires='>=3.8',
        keywords='ai agent langchain langgraph orchestration',
    ) 