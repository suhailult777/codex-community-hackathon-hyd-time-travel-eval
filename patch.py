import re
with open('ui/app.py', 'r', encoding='utf-8') as f:
    content = f.read()
content = re.sub(r'with st\.expander\(".*?Core Research:', 'with st.expander(\"🔬 Core Research:', content)
with open('ui/app.py', 'wb') as f:
    f.write(content.encode('utf-8'))
print('Done')