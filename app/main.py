from app.models.transformer import TransformerModel

if __name__ == "__main__":
    model = TransformerModel()
    answers = model.generate_response(prompt="Hôm nay trời mưa")
    print(answers)