import pandas as pd

# with open(f"papers.csv", 'r', encoding="utf-8") as f:
#     lines = f.readlines()
# for line in lines:
#     print(line)

papers_data = pd.read_csv('papers3.csv')


# papers_data = pd.read_csv(open('papers.csv', encoding='utf-8'))
# print(papers_data.head())


def Gen_Markdown_Single_Year(Year=2023, Year_start=2016, FileName='2023', out_file=True):
    assert Year>=Year_start
    lines = ""

    for i in range(len(papers_data)):
        if Year <=Year_start:
            if papers_data['Year'][i]>Year: continue
        else:
            if not papers_data['Year'][i]==Year: continue

        assert papers_data['Year'][i]
        assert papers_data['Type'][i]
        assert papers_data['Publisher'][i]
        assert papers_data['Title'][i]
        # assert papers_data['Authors'][i] 
        assert papers_data['Paper'][i]

        if not pd.isna(papers_data['Institutions'][i]):
            # print(papers_data['Institutions'][i])
            Tags = "; ".join(["**"+str(papers_data['Type'][i])+"**", str(papers_data['Institutions'][i])])
        else:
            Tags = "**"+str(papers_data['Type'][i])+"**"
        # print(Tags)

        if not pd.isna(papers_data['Code'][i]):
            code_str = f"[[code]]({papers_data['Code'][i]})"
        else:
            code_str = f"*code*"

        if not pd.isna(papers_data['Authors'][i]):
            author_str = f"*{papers_data['Authors'][i]}.* "
        else:
            author_str = f""

        new_line = f"- 【{papers_data['Publisher'][i]}】{papers_data['Title'][i]}. {author_str}[[paper]]({papers_data['Paper'][i]}) {code_str} `Tags:` {Tags}\n"
        # print(new_line)

        lines += new_line

    if out_file:
        with open(f"./Timelines/markdown_{FileName}_part.md", 'w', encoding="utf-8") as f:
            f.writelines(lines)
    return lines

def Gen_Markdown_All_Years(Year_start, Year_end):
    lines = ""
    for Year in range(Year_end, Year_start-1, -1):
        FileName = str(Year) if Year>Year_start else "Earlier"
        CURR_LINES = Gen_Markdown_Single_Year(Year=Year, Year_start=Year_start, FileName=FileName, out_file=True)
        lines += f"### {FileName} <a name='sl_paper_{FileName.lower()}'></a>\n[[Back to TOP]](#table-of-content)\n{CURR_LINES}\n"
        print(Year, "-> finished!")

        with open("./markdown_Timelines_all.md", 'w', encoding="utf-8") as f:
            f.writelines(lines)
        
        print("Gen_Markdown_All_Years")


def Gen_Markdown_Single_Institution(Institution='XMU', out_file=True):
    assert Institution in papers_data.columns

    lines = ""

    for i in range(len(papers_data)):
        # print(papers_data[Institution][i])
        if pd.isna(papers_data[Institution][i]): continue
        
        assert papers_data['Year'][i]
        assert papers_data['Type'][i]
        assert papers_data['Publisher'][i]
        assert papers_data['Title'][i]
        # assert papers_data['Authors'][i] 
        assert papers_data['Paper'][i]

        if not pd.isna(papers_data['Institutions'][i]):
            # print(papers_data['Institutions'][i])
            Tags = "; ".join(["**"+str(papers_data['Type'][i])+"**", str(papers_data['Institutions'][i])])
        else:
            Tags = "**"+str(papers_data['Type'][i])+"**"
        # print(Tags)

        if not pd.isna(papers_data['Code'][i]):
            code_str = f"[[code]]({papers_data['Code'][i]})"
        else:
            code_str = f"*code*"

        if not pd.isna(papers_data['Authors'][i]):
            author_str = f"*{papers_data['Authors'][i]}.* "
        else:
            author_str = f""

        new_line = f"- 【{papers_data['Publisher'][i]}】{papers_data['Title'][i]}. {author_str}[[paper]]({papers_data['Paper'][i]}) {code_str} `Tags:` {Tags}\n"
        # print(new_line)

        lines += new_line

    if out_file:
        with open(f"./Institutions/markdown_{Institution}_part.md", 'w', encoding="utf-8") as f:
            f.writelines(lines)
    return lines

def Gen_Markdown_All_Institutions(Institutions):
    lines = ""
    for Institution in Institutions:
        CURR_LINES = Gen_Markdown_Single_Institution(Institution=Institution, out_file=True)
        lines += f"### {Institution} for AI Sign Language <a name='sl_paper_{Institution.lower()}'></a>\n[[Back to TOP]](#table-of-content)\n{CURR_LINES}\n"
        print(Institution, "-> finished!")

        with open("./markdown_Institutions_all.md", 'w', encoding="utf-8") as f:
            f.writelines(lines)
        
        print("Gen_Markdown_All_Institutions")



if __name__ =="__main__":
    Year_start=2016 # fixed
    Year_end = 2024 # to modify
    Gen_Markdown_All_Years(Year_start, Year_end)

    Institutions = ["XMU", "USTC", "ZJU", "THU", "Germany-UK"]
    Gen_Markdown_All_Institutions(Institutions)
    
    

