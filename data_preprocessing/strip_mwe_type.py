import os
import pandas as pd
import ast

def replace_after_colon(lst):
    lst = ast.literal_eval(lst)
    new = []
    for item in lst:
        if ':' in item:
            # split at colon, keep text before it + "MWE"
            left = item.split(':', 1)[0]
            new.append(f"{left}:MWE")
        else:
            new.append(item)
    return new

def strip_mwe_type(cupt_path, output_path, save=True):
    
    df = pd.read_csv(cupt_path)

    df["MWE_stripped"] = df['PARSEME:MWE'].apply(replace_after_colon)

    # print(df[["PARSEME:MWE", "MWE_stripped"]].head(2))

    # org = df["PARSEME:MWE"].tolist()
    # stripped = df["MWE_stripped"].tolist()

    # print("Original MWE annotations vs Stripped MWE annotations:")
    # for o, s in zip(org[:5], stripped[:5]):
    #     print(f"{o}  -->  {s}")

    if save:
        df.to_csv(output_path, index=False, encoding="utf-8")
    else:
        pass

if __name__ == "__main__":

    processed_data_dir = "/mnt/parscratch/users/acq22zm/parseme_2/data_processed_dev/"

    output_csv_dir = "/mnt/parscratch/users/acq22zm/parseme_2/data_processed_stripped_dev/"


    subtasks = ["subtask1", "subtask1_trial"]


    for subtask in subtasks:
        languages = os.listdir(os.path.join(processed_data_dir, subtask))
        print(f"Processing subtask: {subtask} with languages: {languages}")

        for lang in languages:
            lang_dir = os.path.join(processed_data_dir, subtask, lang)
            output_lang_dir = os.path.join(output_csv_dir, subtask, lang)
            os.makedirs(output_lang_dir, exist_ok=True)

            files = os.listdir(lang_dir)
            for file in files:
                if file.endswith(".csv"):
                    cupt_path = os.path.join(lang_dir, file)
                    # csv_filename = file.replace(".cupt", ".csv")
                    csv_path = os.path.join(output_lang_dir, file)

                    print(f"Converting {cupt_path} to {csv_path}...")
                    strip_mwe_type(cupt_path, csv_path)


