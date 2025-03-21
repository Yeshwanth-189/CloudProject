from locust import HttpUser, task, between

class ResNetLoadTest(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self):
        with open("sample.jpg", "rb") as img:
            self.client.post("/predict/", files={"file": ("sample.jpg", img, "image/jpeg")})
