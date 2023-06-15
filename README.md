# mlops-diabetes-prediction
# Aanmaken Service Principal
az ad sp create-for-rbac -n "MLOps-SP-BD" --role Contributor --scopes /subscriptions/412afafd-c7eb-488d-b2fc-cc920cc0087a/resourceGroups/mlops-bart

{
  "appId": "c158d57a-cac4-4e81-881b-200a600ba756",
  "displayName": "MLOps-SP-BD",
  "password": "k0m8Q~5rI~3LuuV5CmeO8ynu0leKZcKScTq-RbE7",
  "tenant": "4ded4bb1-6bff-42b3-aed7-6a36a503bf7a"
}

# set default workspace
az configure --defaults group=mlops-bart workspace=diabetes-prediction

# docker build vanuit api folder
docker compose up --build


# kubernetes
Aanmaken token: kubectl -n kubernetes-dashboard create token admin-user

kubectl proxy

link:
http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/


kubectl create namespace diabetes-prediction
kubectl apply -f diabetes-prediction-service.yaml
kubectl apply -f diabetes-prediction-deployment.yaml

# helm

vanuit de folder mlops-diabetes-prediction

helm install diabetes-prediction ./helm

# wissen van een chart
helm delete <chart-name>

of

helm uninstall <chart-name>

# local runner
token: 
AYTVNIY7JMBBIP2PTIKJSX3ERMXHI

C:\actions-runner\run.cmd