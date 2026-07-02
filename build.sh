#!/usr/bin/env bash

#Set version 
VERSION="0.4.0"

pip install --upgrade pip
pip install --user --upgrade setuptools==65.5.1 wheel distutils
pip install "urllib3>=1.25.4,<1.27"
pip install twine
pip install wheel
pip install build

export TWINE_USERNAME=${NEXUS_USER}
export TWINE_PASSWORD=${NEXUS_PASSWORD}

mkdir -p sdk/agent_builder/

for folder in base utils llm_client handlers mcp_client storage prebuilt_tasks; do
    cp -r $folder sdk/agent_builder/
done

# Create __init__.py file to make the agent_builder module a package
touch sdk/agent_builder/__init__.py

#Create temporary copies and update the version variable in them 
cp sdk/setup.py sdk/setup.py.temp
cp sdk/pyproject.toml sdk/pyproject.toml.temp
sed "s/{MODULE_VERSION_VAR}/$VERSION/g" sdk/setup.py.temp > sdk/setup.py
sed "s/{MODULE_VERSION_VAR}/$VERSION/g" sdk/pyproject.toml.temp > sdk/pyproject.toml

echo "Building agent_builder package..."
cd sdk
if python3 -m build --no-isolation; then
    # Upload to Nexus only if build was successful
    echo "Uploading to Nexus repository..."
    if [ -d "dist" ] && [ "$(ls -A dist)" ]; then
        python3 -m twine upload --verbose --repository-url https://prod-nexus.sprinklr.com/nexus/repository/spypi/ dist/*
        echo "Build completed successfully for agent_builder"
    else
        echo "ERROR: Build completed but no distribution files found"
    fi
else
    echo "ERROR: Build failed"
fi

#restoring original files
cd ..
mv sdk/setup.py.temp sdk/setup.py
mv sdk/pyproject.toml.temp sdk/pyproject.toml

# Clean up build artifacts and temporary files
rm -rf sdk/dist/
rm -rf sdk/build/
rm -rf sdk/*.egg-info/
rm -rf sdk/agent_builder/
