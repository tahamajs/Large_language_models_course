import json

# Read the notebook
with open('text2sql.ipynb', 'r') as f:
    nb = json.load(f)

# Find the cell with router_graph_builder
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'router_graph_builder = StateGraph(RouterGraphState)' in ''.join(cell['source']):
        # Insert the analyser_node function before router_graph_builder
        insert_idx = 0
        for j, line in enumerate(cell['source']):
            if 'router_graph_builder = StateGraph(RouterGraphState)' in line:
                insert_idx = j
                break
        
        analyser_code = [
            'def analyser_node(state: RouterGraphState, llm: Any) -> dict:\n',
            '    """Real difficulty analysis using LLM"""\n',
            '    print(f"ANALYZER: Analyzing \'{state[\'input_question\']}\'")\n',
            '    \n',
            '    prompt = ChatPromptTemplate.from_messages([\n',
            '        SystemMessage(content="""You are an expert SQL difficulty analyzer. Analyze the given question and schema to determine difficulty level.\n',
            '        - \'simple\': Basic SELECT, single table, simple WHERE conditions\n',
            '        - \'moderate\': Multiple tables, JOINs, basic aggregations\n',
            '        - \'challenging\': Complex JOINs, subqueries, advanced aggregations, window functions\n',
            '        \n',
            '        Respond with only one word: simple, moderate, or challenging"""),\n',
            '        HumanMessage(content=f"Question: {state[\'input_question\']}\n\nSchema: {state[\'input_schema\']}")\n',
            '    ])\n',
            '    \n',
            '    try:\n',
            '        chain = prompt | llm\n',
            '        response = chain.invoke({})\n',
            '        difficulty = response.content.strip().lower() if hasattr(response, \'content\') else str(response).strip().lower()\n',
            '        \n',
            '        if difficulty not in [\'simple\', \'moderate\', \'challenging\']:\n',
            '            difficulty = \'moderate\'  \n',
            '            \n',
            '        print(f"ANALYZER: Determined difficulty as \'{difficulty}\'")\n',
            '        return {\n',
            '            "predicted_difficulty": difficulty,\n',
            '            "difficulty_reasoning": f"LLM analyzed and determined difficulty as {difficulty}"\n',
            '        }\n',
            '    except Exception as e:\n',
            '        print(f"ANALYZER ERROR: {e}")\n',
            '        return {\n',
            '            "predicted_difficulty": "moderate",\n',
            '            "difficulty_reasoning": f"Error in analysis, defaulting to moderate: {e}"\n',
            '        }\n',
            '\n'
        ]
             '        \n',
            '        if difficulty not in [\'simple\', \'m'source'][insert_idx:]
        break

# Write back
with open('text2sql.ipynb', 'w'            '            difficulty = \'moderate\'  \n',
           unction to notebook')
E   
