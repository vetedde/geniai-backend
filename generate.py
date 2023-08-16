import model.maag_genai_model as mg
class Model:
    async def predict(self, prompt):
        image_path = mg.generate_image(prompt)
        return image_path
    
model = Model()

