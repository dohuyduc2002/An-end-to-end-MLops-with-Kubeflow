cd script

# python dataloader.py
python preprocess_and_push.py
python train_and_register.py

cd ..

python pipeline.py
python client.py 

# # kubectl logs -n kubeflow-user-example-com artifact-based-data-preprocessing-pipeline-6lgc5-system-container-driver-4101061331 --all-containers=true
# # substitute the real driver‑pod name
# kubectl logs artifact-based-preprocess-train-pipeline-rpjdm-system-container-driver-4265435456 -n kubeflow-user-example-com --all-containers=true
# kubectl logs preprocess-train-l2f84-system-container-impl-2966834149  -n kubeflow-user-example-com --all-containers=true
