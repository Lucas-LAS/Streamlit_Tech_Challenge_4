import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import joblib
from prophet.plot import plot_plotly, plot_components_plotly

# Configurar o layout de colunas para o menu superior


# Configurar a barra lateral com menu de seleção
st.set_page_config(layout="wide")
st.sidebar.title("Tech Challenge 4 | FIAP")

sidebar_option = st.sidebar.selectbox(
    'Selecione uma opção',
    ('Introdução', 'História Sobre o Preço', 'Análise descritiva', 'Modelo de Previsão')
)


st.markdown(
    """
    <style>
    .stMarkdown a {
        display: inline-block;
        color: white !important;
        text-decoration: none !important;
        white-space: nowrap;
        margin-right: 20px; /* Espaçamento entre os links */
   } 
    .selected-link::after {
        content: '';
        display: block;
        position: relative;
        bottom: -2px; /* Ajuste para a posição da linha */
        width: 100%;
        height: 1px; /* Espessura da linha */
        background-color: white; /* Cor da linha */
    }
        .custom-title {
        color: blue; /* Defina a cor desejada */
    }


    
    </style>
    """,
    unsafe_allow_html=True
)




# Conteúdo da Página Home
if  sidebar_option == "Introdução":
    
    st.markdown("<h1 class='custom-title'>Introdução</h1>", unsafe_allow_html=True)
    st.markdown("<a id='Introdução'></a>", unsafe_allow_html=True)
    st.write("Na introdução deste documento, abordamos alguns temas fundamentais para compreender o projeto, incluindo o conceito de petróleo Brent, a sigla EIA (Agência de Informação de Energia dos Estados Unidos) e o papel do IPEA (Instituto de Pesquisa Econômica Aplicada).")
    #st.markdown("<p style='color: blue;'>Este é outro texto com cor personalizada.</p>", unsafe_allow_html=True)
    st.divider()



    st.markdown("<h3 style='text-align: center;'>Petróleo Brent: Uma Visão Geral</h3>", unsafe_allow_html=True)

    # Texto sobre o petróleo Brent
    st.markdown("""
    O petróleo Brent é uma referência global essencial para os preços do petróleo bruto, utilizado como padrão para determinar os valores de compra e venda em escala mundial. Extraído do Mar do Norte, o Brent é caracterizado como um tipo de petróleo leve e doce, reconhecido por sua alta qualidade. Amplamente negociado nos mercados internacionais de commodities, seu preço é influenciado por diversos fatores, como oferta e demanda globais, eventos geopolíticos, políticas de produção da OPEP e condições econômicas mundiais.

    A relevância do petróleo Brent como padrão internacional deriva de sua ampla disponibilidade e consistente qualidade, tornando-o uma referência confiável para transações comerciais e contratos futuros. Os preços do Brent são frequentemente utilizados como indicadores-chave da saúde e estabilidade da economia global, impactando setores variados, desde transporte e energia até alimentação e manufatura. Consequentemente, variações significativas nos preços do petróleo Brent podem ter repercussões substanciais nas economias nacionais e na política internacional.
    """)

    st.divider()


    st.markdown("<h3 style='text-align: center;'>Instituto de Pesquisa Econômica Aplicada (IPEA)</h3>", unsafe_allow_html=True)

    # Texto sobre o IPEA
    st.markdown("""
    O Instituto de Pesquisa Econômica Aplicada (IPEA) é uma instituição governamental brasileira vinculada ao Ministério da Economia, dedicada à produção de pesquisas e estudos de alta qualidade em economia e políticas públicas. Fundado em 1964, o IPEA desempenha um papel crucial na formulação e avaliação de políticas governamentais, fornecendo análises e recomendações baseadas em evidências para contribuir com o desenvolvimento socioeconômico do Brasil.

    Além de sua relevância na formulação de políticas públicas, o IPEA realiza pesquisas e estudos em diversas áreas, como macroeconomia, mercado de trabalho, saúde, educação, meio ambiente, segurança pública e desenvolvimento regional. Sua produção de conhecimento abrange análises econômicas detalhadas, projeções de cenários futuros, impactos de políticas sociais e avaliações de programas governamentais.

    O IPEA atua como um centro de excelência intelectual, promovendo o debate público e oferecendo subsídios técnicos para a tomada de decisões estratégicas no governo e na sociedade civil. Suas análises são reconhecidas nacional e internacionalmente como fundamentais para entender e enfrentar desafios complexos relacionados ao desenvolvimento econômico e social do Brasil.

    No contexto deste projeto, embora tenham sido inicialmente consultados dados do IPEA, a base final foi obtida diretamente da mesma fonte que eles utilizam, o EIA (Agência de Informação de Energia dos Estados Unidos), garantindo informações atualizadas e relevantes para análises econômicas e políticas públicas.
    """)

    st.divider()


    st.markdown("<h3 style='text-align: center;'>Prophet: Previsão de Séries Temporais</h3>", unsafe_allow_html=True)

    # Texto sobre o Prophet
    st.markdown("""
    O Prophet é uma ferramenta avançada para previsão de dados de séries temporais, baseada em um modelo aditivo que ajusta tendências não lineares, sazonalidades anuais, semanais e diárias, além de considerar efeitos de feriados. Esta abordagem é particularmente eficaz para séries temporais que exibem padrões sazonais fortes e possuem vários anos de dados históricos disponíveis.

    Uma das principais vantagens do Prophet é sua robustez em relação a dados faltantes e mudanças na tendência, além de sua capacidade de lidar bem com valores discrepantes nos dados de entrada. Desenvolvido como um software de código aberto, o Prophet está disponível no GitHub e é mantido pela equipe de cientistas de dados da Meta, garantindo atualizações regulares e suporte contínuo pela comunidade.

    No contexto de previsão de séries temporais, o Prophet se destaca por sua facilidade de uso e pela qualidade das previsões geradas, o que o torna uma ferramenta poderosa para analistas, cientistas de dados e pesquisadores interessados em explorar e prever padrões em dados temporais complexos.
    """)


if  sidebar_option == "História Sobre o Preço":

    st.title("História Sobre o Preço")
    st.markdown("<a id='História Sobre o Preço'></a>", unsafe_allow_html=True)
    st.write("Use o menu superior ou a barra lateral para navegar entre as diferentes visualizações de dados.")
    st.markdown("""
        ### Introdução
        
        Os preços do petróleo são influenciados por uma série de fatores geopolíticos, econômicos e ambientais ao longo do tempo. 
        Desde crises geopolíticas até eventos climáticos extremos, cada acontecimento significativo moldou não apenas os mercados de commodities, 
        mas também a economia global como um todo. Abaixo estão alguns dos eventos mais marcantes que influenciaram os preços do petróleo nas últimas 
        décadas, refletindo mudanças nas políticas de produção, demanda global e estabilidade geopolítica.
        """)

    # Eventos Significativos
    st.markdown("""
        ### Eventos Significativos que Influenciaram os Preços do Petróleo
        
       
        1. **Colapso da União Soviética (1991)**
        2. **Guerra do Golfo (1990-1991)**
        3. **Atentados terroristas nos EUA (2001)**
        4. **Crise financeira global (2007-2008)**
        5. **Primavera Árabe (2010-2012)**
        6. **Desastre do Golfo do México (2010)**
        7. **Guerra do Iraque (2003-2011)**
        8. **Guerra Civil na Líbia (2011)**
        9. **Conflito na Síria (2011~)**
        10. **Sanções internacionais ao Irã (2012)**
        11. **OPEP mantém ritmo de produção (2014)**
        12. **Grande produção e baixa demanda (2015)**
        13. **Acordo Nuclear Irã-P5+1 (2015)**
        14. **Pandemia de COVID-19 (2020-2023)**
        15. **Recuperação econômica pós-COVID (2021~)**
        16. **Conflito Rússia-Ucrânia (2022~)**
        17. **Crise política na Venezuela (2002 e 2019~)**
        
        Esses eventos exemplificam como mudanças políticas, econômicas e sociais globais podem ter impactos profundos nos mercados de energia, 
        afetando não apenas os preços do petróleo, mas também a estabilidade econômica e política em escala internacional.
        """)
    

    df =pd.read_csv('Data//preco_petroleo.csv')

    def plot_grafico_evolucao_preco_petroleo():
        def add_ponto_interesse(fig, ponto, text_index, label):
            fig.add_trace(
                go.Scatter(
                    x=[ponto.ds.values[0]],
                    y=[ponto.y.values[0]],
                    mode="markers",
                    text=1,
                    marker=dict(color="red", size=10, line=dict(color="white", width=1)),
                    name=label,
                )
            )
            fig.add_annotation(
                x=ponto.ds.values[0],
                y=ponto.y.values[0] + 4,
                text=text_index,
                showarrow=False,
                font=dict(color="white", size=10),
                bgcolor="red",
                borderwidth=1,
                bordercolor="white",
            )
            fig.add_shape(
                type="line",
                x0=ponto.ds.values[0],
                x1=ponto.ds.values[0],
                y0=min(df.y),
                y1=max(df.y),
                line=dict(color="red", width=1, dash="dash")
            )

    
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df.ds, y=df.y, mode="lines", name="Preço do barril de petróleo")
        )
        eventos_chave = [
                ("1991-12-26", 1, "1. Colapso da União Soviética (1991)"),
                ("1990-08-02", 2, "2. Guerra do Golfo (1990-1991)"),
                ("2001-09-11", 3, "3. Atentados terroristas nos EUA (2001)"),
                ("2007-08-01", 4, "4. Crise financeira global (2007-2008)"),
                ("2010-12-20", 5, "5. Primavera Árabe (2010-2012)"),
                ("2010-04-20", 6, "6. Desastre do Golfo do México (2010)"),
                ("2011-02-17", 7, "7. Guerra Civil na Líbia (2011)"),
                ("2011-03-15", 8, "8. Conflito na Síria (a partir de 2011)"),
                ("2012-06-01", 9, "9. Sanções internacionais ao Irã (2012)"),
                ("2014-11-28", 10, "10. OPEP mantém ritmo de produção (2014)"),
                ("2015-01-02", 11, "11. Grande produção e baixa demanda (2015)"),
                ("2015-06-24", 12, "12. Acordo Nuclear Irã-P5+1 (2015)"),
                ("2020-01-30", 13, "13. Pandemia de COVID-19 (2020-2023)"),
                ("2021-07-01", 14, "14. Recuperação econômica pós-COVID (2021-presente)"),
                ("2022-02-24", 15, "15. Conflito Rússia-Ucrânia (2022-presente)"),
                ("2002-04-11", 16, "16. Crise política na Venezuela (2002 e 2019)"),
            ]      
        for data, indice, descricao in eventos_chave:
            add_ponto_interesse(fig, df.query(f'ds == "{data}"'), indice, descricao)

        fig.add_annotation(
            x=0.5,
            y=-0.15,
            xref="paper",
            yref="paper",
            text="<b>Fonte: EIA, 2024</b>",
            showarrow=False,
            font=dict(color="black", size=10),
            bgcolor="white",
            borderwidth=1,
            bordercolor="black",
        )

        fig.update_layout(
            title="Evolução do preço do barril de petróleo Brent ao longo das decádas (1987 até hoje)",
            xaxis_title="Data",
            yaxis_title="Preço em US$",
            height=640,
        )

        st.plotly_chart(fig, use_container_width=True)

    plot_grafico_evolucao_preco_petroleo()

    texto_explicacao = """
Desse conjunto de 17 períodos de eventos que moldaram os preços do petróleo ao longo das décadas, optou-se por focar nos três períodos que se destacam por sua significativa volatilidade. Cada um desses períodos não apenas testemunhou picos dramáticos de aumento e quedas vertiginosas nos preços do petróleo, mas também refletiu mudanças profundas na geopolítica global e nos padrões de demanda. Esses eventos não apenas impactaram os mercados de commodities, mas também tiveram repercussões econômicas e políticas em escala global.
"""
    st.markdown(texto_explicacao)

    periodo1_inicio = "2007-08-01"
    periodo1_fim = "2008-12-31"

    periodo2_inicio = "2015-01-02"
    periodo2_fim ="2015-12-31"

    periodo3_inicio = "2022-01-01"
    periodo3_fim = "2024-01-01"


    # Filtrando dados para cada período
    df_periodo1 = df[(df['ds'] >= periodo1_inicio) & (df['ds'] <= periodo1_fim)]
    df_periodo2 = df[(df['ds'] >= periodo2_inicio) & (df['ds'] <= periodo2_fim)]
    df_periodo3 = df[(df['ds'] >= periodo3_inicio) & (df['ds'] <= periodo3_fim)]

    # Função para adicionar retângulos destacando os maiores e menores valores
    def add_highlight_rectangles(fig, df_periodo, title):
        max_value = df_periodo['y'].max()
        min_value = df_periodo['y'].min()
        max_index = df_periodo['y'].idxmax()
        min_index = df_periodo['y'].idxmin()
        
        # Adicionando retângulo para o máximo
        fig.add_shape(
            type="rect",
            x0=df_periodo.loc[max_index, 'ds'], y0=min(df['y']), 
            x1=df_periodo.loc[max_index, 'ds'], y1=max(df['y']),
            line=dict(color="red", width=1, dash="dash"),
            fillcolor="red",
            opacity=0.3,
            layer="below",
            name=f'Máximo ({max_value})'
        )
        
        # Adicionando retângulo para o mínimo
        fig.add_shape(
            type="rect",
            x0=df_periodo.loc[min_index, 'ds'], y0=min(df['y']), 
            x1=df_periodo.loc[min_index, 'ds'], y1=max(df['y']),
            line=dict(color="blue", width=1, dash="dash"),
            fillcolor="blue",
            opacity=0.3,
            layer="below",
            name=f'Mínimo ({min_value})'
        )
        # Anotação explicativa para os retângulos
        fig.add_annotation(
            x=df_periodo.loc[max_index, 'ds'],
            y=max(df['y']) * 0.95,
            xref="x",
            yref="y",
            text=f'Máximo valor: {max_value}',
            showarrow=True,
            arrowhead=2,
            ax=20,
            ay=-30
        )
        
        fig.add_annotation(
            x=df_periodo.loc[min_index, 'ds'],
            y=max(df['y']) * 0.95,
            xref="x",
            yref="y",
            text=f'Mínimo valor: {min_value}',
            showarrow=True,
            arrowhead=2,
            ax=-20,
            ay=-30
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Data',
            yaxis_title='Preço em US$'
        )

    # Criando os gráficos de linha para cada período
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_periodo1['ds'], y=df_periodo1['y'], mode='lines', name='Período 1'))
    add_highlight_rectangles(fig1, df_periodo1, 'Crise financeira global (2007-2008)')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_periodo2['ds'], y=df_periodo2['y'], mode='lines', name='Período 2'))
    add_highlight_rectangles(fig2, df_periodo2, 'Grande produção e baixa demanda (2015)')

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_periodo3['ds'], y=df_periodo3['y'], mode='lines', name='Período 3'))
    add_highlight_rectangles(fig3, df_periodo3, 'Conflito Rússia-Ucrânia (2022~)')

    texto_periodo1 = """
    Durante a crise financeira global de 2007-2008, desencadeada pelo colapso do mercado imobiliário dos EUA, houve uma redução drástica na demanda global por petróleo. A crise levou a uma recessão econômica global, resultando em cortes significativos na produção industrial e um declínio no consumo de energia. Isso causou uma queda nos preços do petróleo, à medida que a oferta superava a demanda.
    """

    texto_periodo2 = """
    Em 2015, o mercado global de petróleo enfrentou um período de grande produção por parte dos principais produtores, como os membros da OPEP e os Estados Unidos. Paralelamente, a demanda global começou a desacelerar devido à desaceleração econômica em grandes economias consumidoras, como a China. Esse desequilíbrio entre oferta abundante e demanda reduzida resultou em uma queda acentuada nos preços do petróleo.
    """

    texto_periodo3 = """
    O conflito entre Rússia e Ucrânia, que se intensificou a partir de 2022, teve repercussões significativas nos mercados globais de energia. A escalada das tensões geopolíticas na região, um importante corredor de trânsito de gás natural e petróleo, gerou preocupações quanto à segurança dos suprimentos energéticos. Isso levou a volatilidades nos preços do petróleo, com investidores reagindo a potenciais interrupções na produção e no transporte de energia.
    """

    # Exibindo os gráficos no Streamlit
    st.markdown("---")
    st.subheader("Crise Financeira Global (2007-2008)")
    st.markdown(texto_periodo1)

    col1, col2 = st.columns([2, 1])  # Coluna para os gráficos e coluna para os boxplots

    
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
        # st.plotly_chart(fig2, use_container_width=True)
        # st.plotly_chart(fig3, use_container_width=True)

    with col2:
        # Criando os boxplots para cada período
        fig_boxplot1 = go.Figure()
        fig_boxplot1.add_trace(go.Box(y=df_periodo1['y'], name='(2007-2008)'))
        fig_boxplot1.update_layout(title='Boxplot - Crise financeira global (2007-2008)',
                                yaxis_title='Preço em US$')

        fig_boxplot2 = go.Figure()
        fig_boxplot2.add_trace(go.Box(y=df_periodo2['y'], name='2015'))
        fig_boxplot2.update_layout(title='Boxplot - Grande produção e baixa demanda (2015)',
                                yaxis_title='Preço em US$')

        fig_boxplot3 = go.Figure()
        fig_boxplot3.add_trace(go.Box(y=df_periodo3['y'], name='2022~'))
        fig_boxplot3.update_layout(title='Boxplot - Conflito Rússia-Ucrânia (2022~)',
                                yaxis_title='Preço em US$')

        st.plotly_chart(fig_boxplot1, use_container_width=True)
    st.markdown("---")
    st.subheader("Grande Produção e Baixa Demanda (2015)")
    st.markdown(texto_periodo2)

    col1, col2 = st.columns([2, 1])  # Coluna para os gráficos e coluna para os boxplots
    with col1:
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.plotly_chart(fig_boxplot2, use_container_width=True)

    st.markdown("---")
    st.subheader("Conflito Rússia-Ucrânia (2022~)")
    st.markdown(texto_periodo3)

    col1, col2 = st.columns([2, 1])  # Coluna para os gráficos e coluna para os boxplots
    with col1:
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.plotly_chart(fig_boxplot3, use_container_width=True)


 





if  sidebar_option == "Análise descritiva":

    st.title("Análise descritiva")
    st.markdown("<a id='Análise descritiva'></a>", unsafe_allow_html=True)
  
  
    df =pd.read_csv('Data//preco_petroleo.csv')

   # Título da página
    st.title('Análise Descritiva da Base de Dados')

    # Subtítulo e explicação inicial
    st.header('Visão Geral')
    st.markdown("""
    Esta página apresenta uma análise descritiva da base de dados, focando nos dados de séries temporais contendo informações sobre datas (`ds`) e valores (`y`).
    """)
    st.markdown("---")

    # Estatísticas descritivas
    st.header('Estatísticas Descritivas')
    st.write(df.describe())
    st.markdown("---")
    # Gráfico de Linha dos Dados Temporais
    st.header('Gráfico de Linha dos Dados Temporais')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Dados Temporais'))
    fig.update_layout(title='Série Temporal dos Dados',
                    xaxis_title='Data',
                    yaxis_title='Valor')
    st.plotly_chart(fig, use_container_width=True)
    # Explicação do Gráfico de Linha
    st.markdown("""
    O gráfico acima mostra a série temporal dos dados, exibindo a variação dos valores ao longo do tempo. Podemos observar uma tendência de crescimento geral nos dados.
    """)
    st.markdown("---")

    # Histograma dos Dados
    st.header('Histograma dos Dados')
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=df['y'], nbinsx=20))
    fig_hist.update_layout(title='Distribuição dos Dados',
                        xaxis_title='Valor',
                        yaxis_title='Frequência')
    st.plotly_chart(fig_hist, use_container_width=True)

    # Explicação do Histograma
    st.markdown("""
    O histograma mostra a distribuição dos dados, ajudando a visualizar a frequência com que diferentes valores ocorrem na base de dados.
    """)
    st.markdown("---")

    # Análise de Tendência Temporal
    st.header('Análise de Tendência Temporal')
    df['rolling_mean'] = df['y'].rolling(window=30, min_periods=1).mean()
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Dados Temporais'))
    fig_trend.add_trace(go.Scatter(x=df['ds'], y=df['rolling_mean'], mode='lines', name='Média Móvel (30 dias)'))
    fig_trend.update_layout(title='Análise de Tendência Temporal',
                            xaxis_title='Data',
                            yaxis_title='Valor')
    st.plotly_chart(fig_trend, use_container_width=True)

    # Explicação da Análise de Tendência Temporal
    st.markdown("""
    A análise de tendência temporal inclui a média móvel dos dados, destacando a direção geral dos valores ao longo do tempo. Isso ajuda a identificar padrões de longo prazo nos dados.
    """)
    st.markdown("---")

    # Decomposição de Séries Temporais
    st.header('Decomposição de Séries Temporais')
    result = seasonal_decompose(df['y'], model='additive', period=30)
    fig_decompose = go.Figure()
    fig_decompose.add_trace(go.Scatter(x=df['ds'], y=result.trend, mode='lines', name='Tendência'))
    fig_decompose.add_trace(go.Scatter(x=df['ds'], y=result.seasonal, mode='lines', name='Sazonalidade'))
    fig_decompose.add_trace(go.Scatter(x=df['ds'], y=result.resid, mode='lines', name='Resíduos'))
    fig_decompose.update_layout(title='Decomposição de Séries Temporais',
                                xaxis_title='Data')
    st.plotly_chart(fig_decompose, use_container_width=True)

    # Explicação da Decomposição de Séries Temporais
    st.markdown("""
    A decomposição de séries temporais separa os dados em três componentes principais: tendência, sazonalidade e resíduos. Isso ajuda a entender melhor a estrutura dos dados e os padrões subjacentes.
    """)

if  sidebar_option == "Modelo de Previsão":

    st.title("Modelo de Previsão")
    st.markdown("<a id='Modelo de Previsão'></a>", unsafe_allow_html=True)
    # Título do aplicativo
    st.title('Previsão de Preços do Petróleo com Prophet')

    # Texto de introdução
    st.write("""
    Prever o preço do barril de petróleo é crucial para diversos setores, e o Prophet é uma ferramenta poderosa nesse desafio. Desenvolvido pela Meta, o Prophet destaca-se pela sua facilidade de uso e capacidade de modelar padrões sazonais complexos. Ele utiliza um modelo aditivo em que tendências não lineares são ajustadas com base em sazonalidades diárias, semanais e anuais, tornando-o ideal para dados que apresentam múltiplos padrões sazonais.

    Uma das grandes vantagens do Prophet é a sua capacidade de lidar com dados ausentes e mudanças abruptas nas séries temporais, o que é comum no mercado de petróleo devido a eventos geopolíticos e econômicos. Além disso, o Prophet permite a inclusão de variáveis externas (ou regressoras), que podem melhorar ainda mais a precisão das previsões ao considerar fatores externos que impactam os preços do petróleo.

    A interface amigável e a facilidade de ajuste de parâmetros fazem do Prophet uma ferramenta acessível tanto para analistas com pouca experiência em modelagem de séries temporais quanto para especialistas. Ele proporciona uma abordagem robusta para compreender e prever as flutuações no mercado de energia, fornecendo insights essenciais para decisões estratégicas e planejamento futuro.
    """)

    # Carregar o modelo Prophet salvo
    model_path = 'prophet_modelo.joblib'  # Altere para o caminho do seu modelo salvo
    model = joblib.load(model_path)

  

  
    df = pd.read_csv('previsao.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    st.write(df.head())

     # Fazer previsões
    st.write("### Visualização dos primeiros registros do conjunto de dados")
    st.write("Aqui estão os primeiros registros do conjunto de dados carregado:")
    st.write(df.head())

    # Fazer previsões
    st.write("### Fazer previsões")
    st.write("Usando o modelo Prophet para prever os preços do petróleo para os próximos 20 dias:")
    future = model.make_future_dataframe(periods=20)
    forecast = model.predict(future)
    
    st.write("### Explicação dos Campos da Previsão")
    st.write("""
    - **ds**: A data da previsão.
    - **yhat**: O valor previsto pelo modelo para essa data.
    - **yhat_lower**: O limite inferior do intervalo de confiança da previsão.
    - **yhat_upper**: O limite superior do intervalo de confiança da previsão.
    """)
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Visualizar previsões
    st.write("### Visualizar previsões")
    st.write("O gráfico abaixo mostra as previsões de preços do petróleo geradas pelo modelo Prophet. As linhas representam os valores previstos, enquanto as áreas sombreadas mostram os intervalos de confiança:")
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1)

    # Plotar componentes
    st.write("### Componentes das previsões")
    st.write("O gráfico abaixo detalha os componentes das previsões feitas pelo modelo Prophet, incluindo a tendência geral, a sazonalidade anual e os efeitos semanais:")
    fig2 = plot_components_plotly(model, forecast)
    st.plotly_chart(fig2)

    # Solicitar períodos do usuário
    st.write("### Ajuste de Períodos para Previsão")
    periods_input = st.number_input('Digite o número de períodos para previsão (dias):', min_value=1, value=30)

    if st.button('Gerar Novas Previsões'):
        st.write(f"### Previsões para os próximos {periods_input} dias")
        future = model.make_future_dataframe(periods=periods_input)
        forecast = model.predict(future)
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        st.write("### Visualizar novas previsões")
        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1)

        st.write("### Novos componentes das previsões")
        fig2 = plot_components_plotly(model, forecast)
        st.plotly_chart(fig2)


with st.sidebar:
        st.divider()
        st.subheader("Aluno")
        st.text("Lucas Azevedo Silva")
        st.text("RM -352959  ")
        st.text("Grupo - 41")

        st.divider()

  

        st.subheader("Executando localmente")
        st.code(body="streamlit run .\streamlit_app.py", language="shell")

        st.divider()

        st.subheader("Repositórios do projeto")
        st.link_button(
            "Repositório Streamlit",
            "",
            help=None,
            type="secondary",
            disabled=False,
            use_container_width=False,
        )
        st.link_button(
            "Repositório Jupyter Notebook",
            "",
            help=None,
            type="secondary",
            disabled=False,
            use_container_width=False,
        )