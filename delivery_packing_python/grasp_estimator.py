import requests

grasp_options = {'max_distance':0.01, 'min_nb_grasp':3, 'max_try':3, 'min_confidence':0.8, 'visualize':"filtred"}



class GraspEstimator:
    def __init__(self):
        pass

    def init_distant_container(self):
        try:
            response = requests.post('http://localhost:8001/init', json={'ex_param':[], "options":grasp_options}).json()
        except Exception as e:
            print("Error connecting to the grasp estimation module. Make sure the docker container is running.")
            raise e
        