import pandas as pd
import os

def cupt_to_sentence_csv(cupt_path, lang, csv_path):
    sentences = []
    columns = []
    sent_meta = {"sent_id": None, "sentence_text": None}
    sent_tokens = []
    
    # ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC PARSEME:MWE

    with open(cupt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Sentence boundary
            if not line:
                if sent_tokens:
                    # Flatten token fields into lists
                    sent_row = {"sent_id": sent_meta["sent_id"],
                                "sentence_text": sent_meta["sentence_text"]}
                    for c in columns:
                        sent_row[c] = [tok[c] for tok in sent_tokens]
                    sentences.append(sent_row)
                sent_tokens = []
                sent_meta = {"sent_id": None, "sentence_text": None}
                continue

            # Comments
            if line.startswith("#"):
                if line.startswith("# global.columns"):
                    columns = line.split("=", 1)[1].strip().split()
                elif line.startswith("# text ="):
                    sent_meta["sentence_text"] = line.split("=", 1)[1].strip()
                elif line.startswith("# source_sent_id ="):
                    sent_meta["sent_id"] = line.split("=", 1)[1].strip()
                continue

            # Token line
            parts = line.split("\t")
            if len(parts) != len(columns):
                # skip multiword range lines like "2-3"
                continue
            sent_tokens.append(dict(zip(columns, parts)))

    # handle last sentence (if file doesn't end with blank line)
    if sent_tokens:
        sent_row = {"sent_id": sent_meta["sent_id"],
                    "sentence_text": sent_meta["sentence_text"]}
        for c in columns:
            sent_row[c] = [tok[c] for tok in sent_tokens]
        sentences.append(sent_row)

    # df
    df = pd.DataFrame(sentences)
    df["lang"] = lang
    df.to_csv(csv_path, index=False, encoding="utf-8")

    return df





if __name__ == "__main__":

    #one file example
    
    # df = cupt_to_sentence_csv("/mnt/parscratch/users/acq22zm/parseme_2/sharedtask-data/2.0/subtask1_trial/EN/trial.train.cupt", "")

    # # print(df.head(20))
    # print(df.columns)

    # print(df["FORM"].head(1))
    # print(df["PARSEME:MWE"].head(1))


    # index = 1

    # form = df.loc[index, 'FORM']
    # print(form) 

    # parseme_mwe = df.loc[index, 'PARSEME:MWE']
    # print(parseme_mwe)

    # sharedtask_data_dir = "/mnt/parscratch/users/acq22zm/parseme_2/sharedtask-data/2.0/"
    # output_csv_dir = "/mnt/parscratch/users/acq22zm/parseme_2/data_processed/"


    # subtasks = ["subtask1", "subtask1_trial"]

    # for subtask in subtasks:
    #     languages = os.listdir(os.path.join(sharedtask_data_dir, subtask))
    #     print(f"Processing subtask: {subtask} with languages: {languages}")

    #     for lang in languages:
    #         lang_dir = os.path.join(sharedtask_data_dir, subtask, lang)
    #         output_lang_dir = os.path.join(output_csv_dir, subtask, lang)
    #         os.makedirs(output_lang_dir, exist_ok=True)

    #         files = os.listdir(lang_dir)
    #         for file in files:
    #             if file.endswith(".cupt"):
    #                 cupt_path = os.path.join(lang_dir, file)
    #                 csv_filename = file.replace(".cupt", ".csv")
    #                 csv_path = os.path.join(output_lang_dir, csv_filename)

    #                 print(f"Converting {cupt_path} to {csv_path}...")
    #                 cupt_to_sentence_csv(cupt_path, lang, csv_path)

    # #note: subtask1 language: ['SV', 'SL', 'NL', 'EL', 'EGY', 'KA', 'UK', 'FR', 'SR', 'GRC', 'HE', 'FA', 'PT', 'LV', 'tools', 'RO', 'PL', 'JA']
    # #note: subtask1_trial language: ['FR', 'EN']

    sharedtask_data_dir = "/mnt/parscratch/users/acq22zm/parseme_2/sharedtask-data/2.0/"
    output_csv_dir = "/mnt/parscratch/users/acq22zm/parseme_2/data_processed_dev/"


    subtasks = ["subtask1", "subtask1_trial"]

    for subtask in subtasks:
        languages = os.listdir(os.path.join(sharedtask_data_dir, subtask))
        print(f"Processing subtask: {subtask} with languages: {languages}")

        for lang in languages:
            lang_dir = os.path.join(sharedtask_data_dir, subtask, lang)
            output_lang_dir = os.path.join(output_csv_dir, subtask, lang)
            os.makedirs(output_lang_dir, exist_ok=True)

            files = os.listdir(lang_dir)
            for file in files:
                if file.endswith("dev.cupt"):
                    cupt_path = os.path.join(lang_dir, file)
                    csv_filename = file.replace(".cupt", ".csv")
                    csv_path = os.path.join(output_lang_dir, csv_filename)

                    print(f"Converting {cupt_path} to {csv_path}...")
                    cupt_to_sentence_csv(cupt_path, lang, csv_path)

