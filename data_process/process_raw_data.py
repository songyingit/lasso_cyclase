import pandas as pd
import re

def mutate_precursor(mutation, seq):
    seq = seq.copy()
    if mutation == "WT":
        return "".join(seq)
    for mut in mutation.split(","):
        if "del" in mut:
            pos = int(''.join(filter(str.isdigit, mut)))
            seq[pos-1] = ""
        elif "ins" in mut:
            aa = mut[4]
            pos = int(''.join(filter(str.isdigit, mut)))
            seq[pos-2] = seq[pos-2] + aa
        else:
            old = mut[0]
            pos = int(mut[1:-1])
            new = mut[-1]
            if seq[pos-1] != old:
                raise ValueError(f"Mismatch at {pos}: expected {old}, found {seq[pos-1]}")
            seq[pos-1] = new
    return "".join(seq)

def mutate_fusc(mutation, seq):
    seq = seq.copy()
    if mutation == "WT":
        return "".join(seq)
    for mut in mutation.split("/"):
        old = mut[0]
        new = mut[-1]
        pos = int(mut[1:-1])
        if seq[pos-1] != old:
            raise ValueError(f"Mismatch at {pos}: expected {old}, found {seq[pos-1]}")
        seq[pos-1] = new
    return "".join(seq)

if  __name__ == "__main__":

      raw_data = pd.read_excel("../raw_data/FusC_variants.xlsx", sheet_name='Sheet1')
      raw_df = pd.DataFrame(raw_data)
      FusA_WT_seq = 'WYTAEWGLELIFVFPRFI'
      print("FusA_WT_seq length:", len(FusA_WT_seq))
      FusC_WT_seq = 'MVGCISPYFAVFPDKDVLGQATDRLPAAQTLASHPSGRPWLVGALPADQLLLVEAGERRLAVIGHCSAEPERLRAELAQIDDVAQFDRLARTLDGSFHLVVVVGDQMRIQGSVSGLRRVFHAHVGTARIAADRSDVLAAVLGVSPDPDVLALRMFNGLPYPLSELPPWPGVEHVPAWHYLSLGLHDGRHRVVQWWHPPEAELDVTAAAPLLRTALAGAVDTRTRGGGVVSADLSGGLDSTPLCALAARGPAKVVALTFSSGLDTDDDLRWAKIAHQSFPSVEHVVLSPEDIPGFYAGLDGEFPLLDEPSVAMLSTPRILSRLHTARAHGSRLHMDGLGGDQLLTGSLSLYHDLLWQRPWTALPLIRGHRLLAGLSLSETFASLADRRDLRAWLADIRHSIATGEPPRRSLFGWDVLPKCGPWLTAEARERVLARFDAVLESLEPLAPTRGRHADLAAIRAAGRDLRLLHQLGSSDLPRMESPFLDDRVVEACLQVRHEGRMNPFEFKSLMKTAMASLLPAEFLTRQSKTDGTPLAAEGFTEQRDRIIQIWRESRLAELGLIHPDVLVERVKQPYSFRGPDWGMELTLTVELWLRSRERVLQGANGGDNRS'
      print("FusC_WT_seq length:", len(FusC_WT_seq))

      raw_df["FusA_seq"] = raw_df["Precursor"].apply(lambda x: mutate_precursor(x, list(FusA_WT_seq)))
      raw_df["FusC_seq"] = raw_df["FusC"].apply(lambda x: mutate_fusc(x, list(FusC_WT_seq)))
      raw_df["Substate"] = raw_df["TurnOver"].map({"Y":int(1), "N":int(0), "Y(tiny!)":int(1)})
      processed_df = raw_df[["FusA_seq", "FusC_seq", "Substate"]]
      print("Processed dataframe:", processed_df.info())
      processed_df.to_excel("FusA_FusC_variants_processed.xlsx", index=False)

