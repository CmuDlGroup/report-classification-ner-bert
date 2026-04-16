from train import train_model
from utils.utils import evaluate_models_on_test


if __name__ == "__main__":
    #change the huggingface repository to your own or to your liking for pushing to that repository
    model, tokenizer, dataset = train_model("NASA-AIML/MIKA_SafeAeroBERT", "theophilusowiti/asn-ner-aerobert")

    #you can decide to only do evaluation given that the model is loaded #TODO
    models_to_evaluate = {
    
        "theophilusowiti/asn-ner-aerobert": (model, tokenizer)
    }

    
    test_results_df, all_predictions, all_labels, metric = evaluate_models_on_test(models_to_evaluate, dataset["test"])

    #print evaluation
    print(test_results_df)