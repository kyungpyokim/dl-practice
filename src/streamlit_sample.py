import streamlit_sample as st

st.title('스트림릿 제목')
st.header('헤더')
st.subheader('서브헤더')
st.text('일반 텍스트')
st.markdown('**마크다운 지원** :sparkles:')
st.code("print('Hello World')", language='python')

col1, col2, col3, col4 = st.columns(4)
col1.write('컬럼 1')
col2.write('컬럼 2')
col3.write('컬럼 3')
col4.write('컬럼 4')

with st.expander('펼치기/접기'):
    st.write('관리자 전화번호 : 010-000-0000')
