 
#!/bin/bash
# Enable Jupyter extensions
jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build
jupyter labextension install @pyviz/jupyterlab_pyviz --no-build
jupyter lab build --dev-build=False --minimize=True

# Create directories for student work
mkdir -p /home/jovyan/my_work
mkdir -p /home/jovyan/my_figures

# Make scripts executable
chmod +x lessons/*.py

# Set up matplotlib backend
echo "import matplotlib; matplotlib.use('Agg')" >> /opt/conda/lib/python3.9/site-packages/sitecustomize.py