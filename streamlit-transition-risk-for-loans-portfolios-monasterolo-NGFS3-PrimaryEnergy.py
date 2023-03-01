import streamlit as st
import pandas as pd
import pyam
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

EjDeltaRatio=1
Chi=0.1
RecoveryRate=0.5
year0=2010

header = st.container()
dataset = st.container()
shocks = st.container()
loanvalue = st.container()

@st.cache_data
def get_data():
    pyam.iiasa.set_config("swan", "5w6uwwL2L7eJyW")
    conn = pyam.iiasa.Connection()
    conn.connect('ngfs_phase_3')

    df = conn.query(
        model='*GLOBIOM*',
        region=['*France*','*Germany*'],
        variable='*Primary Energy*'
    )
    
    dff=df.data
    dff=dff[dff['variable'].str.contains('Price') == False]
    dff=dff[dff['year']>=year0]
    dff=dff[(dff['scenario'].str.contains('Below')==True) | (dff['scenario'].str.contains('Current')==True) | (dff['scenario'].str.contains('Delayed')==True)]

    # pivot the primary energy source columns
    dffp = dff.pivot_table(
        values='value',
        index=['scenario','model', 'region','year', 'unit'],
        columns='variable'
        )

    # set the indexes as new columns
    dffp.reset_index(inplace=True)

    #drop the name of the set of columns
    dffp.columns.name=None
    
    return dffp

dffp=get_data()

sectors_columns=['Primary Energy|Biomass', 'Primary Energy|Coal', 'Primary Energy|Gas','Primary Energy|Geothermal', 'Primary Energy|Hydro','Primary Energy|Nuclear', 'Primary Energy|Oil','Primary Energy|Solar','Primary Energy|Wind']
sectors=['Biomass', 'Coal','Gas','Geothermal','Hydro', 'Nuclear', 'Oil','Solar','Wind']
MarketShareSectorsColumns=['MarketShare'+sectors[i] for i in range(len(sectors))]
InitialMarketShareSectorsColumns=['Initial'+MarketShareSectorsColumns[i] for i in range(len(sectors))]
ShocksSectorsColumns=['Shocks'+sectors[i] for i in range(len(sectors))]


# add a new column for the total primary energy, as the sum of the different primary energy of all the sectors
dffp['PrimaryEnergyCalculated'] = dffp[sectors_columns].sum(axis=1)

#add to the inition pandas dataframe, columns for  market shares for each of the sectors of the sector list.
for i in range(len(sectors)):
    dffp[MarketShareSectorsColumns[i]] = dffp[sectors_columns[i]].div(dffp['PrimaryEnergyCalculated'])


# add columns for initial market size for primary energy, depending on the country. This will be used to compute the shocks
for i in range(len(sectors)):
  region_list = dffp['region'].unique()
  
  #extract the list of initial market shares, it has as many elements as there are regions
  initial_market_shares_list = []
  for region in region_list:
    row_of_interest = dffp[(dffp['scenario'].str.contains('Below')==True) & (dffp['year']==year0) & (dffp['region'] == region)]
    initial_market_shares_list.append(row_of_interest[MarketShareSectorsColumns[i]].iloc[0])

  #create a function, you give it the region, it gives you the initial market share in that region (for a given sector)
  mapping=dict(zip(region_list,initial_market_shares_list))

  #create a new column (for a given sector), which gives the initial market shares, depending on the region
  dffp[InitialMarketShareSectorsColumns[i]]=dffp['region'].apply(lambda x: mapping[x])    


# compute the shocks, add them as columns, a column for each sector
for i in range(len(sectors)):
  dffp[ShocksSectorsColumns[i]] = (dffp[MarketShareSectorsColumns[i]]-dffp[InitialMarketShareSectorsColumns[i]]).div(dffp[InitialMarketShareSectorsColumns[i]])


#compute the market shares per sector at year0
InitialMarketSharesDataFrame=dffp[['scenario','year', 'region']+InitialMarketShareSectorsColumns][(dffp['year']==year0) & (dffp['scenario'].str.contains('Below')==True)]
UnpivotedInitialMarketSharesDataFrame =  pd.melt(InitialMarketSharesDataFrame, id_vars=['scenario','year', 'region'])


#Shocks matrix
ShocksMatrix=dffp[['scenario','year', 'region']+ShocksSectorsColumns]




with header:
    st.title('Transition risk for a loan portfolio - Climate stress test')
    #st.text('')

with dataset:
    st.header('Market shares today')
    st.subheader('Chart showing the market sector of different primary energy sources')
    st.markdown('* **This graph:** shows the marketshare of each energy sector in year '+str(year0) )
    InitialMarketSharesPlot = sns.catplot(data=UnpivotedInitialMarketSharesDataFrame, y='variable',x='value', kind='bar',hue='region', height=5, aspect=12/5 )
    st.pyplot(InitialMarketSharesPlot)
    #st.bar_chart(data=UnpivotedInitialMarketSharesDataFrame,x='sectors', y='sectors share today')
    


with shocks:
    st.header('The shocks')
    sel_col, disp_col = st.columns(2)
   
    
    The_sector=sel_col.selectbox("Choose Sector for which you want to  plot the shocks:", options=['Biomass', 'Coal','Gas','Geothermal','Hydro','Solar','Wind', 'Nuclear', 'Oil'])
    
    The_region = disp_col.selectbox('Choose the region:', options=dffp['region'].unique())
    
    
    SchocksPlot = sns.relplot(data=dffp[dffp['region']==The_region], kind='line', x='year', y='Shocks'+The_sector, hue='scenario', height=5, aspect=8/5)
    st.pyplot(SchocksPlot)

    sel_col.text('Here are the models in the dataset:')
    The_country=sel_col.write(dffp['model'].unique())

    

with loanvalue:
    sel_col2, disp_col2 = st.columns(2)
    st.header('Simulation for the value of a loan')
    st.text('Choose your portfolio')
    Chi = 0.01*sel_col2.slider('Chi as a percentage', min_value=0, max_value=10, value=100, step=10)

    FaceValuesOfLoans1=[1000,100000,1000,2000,2000,2000,100,100,1000]
    FaceValuesOfLoans2={'Sectors':sectors, 'Face Values of Loans':FaceValuesOfLoans1}
    FaceValuesOfLoans=pd.DataFrame(FaceValuesOfLoans2)

    #ChangeInDefaultProbMatrix
    ChangeInDefaultProbMatrix =ShocksMatrix.copy()
    ChangeInDefaultProbMatrix[ShocksSectorsColumns] = -EjDeltaRatio*Chi*ShocksMatrix[ShocksSectorsColumns]

    # Compute the change in values of loans. This should be modified since in our probability matrix we have probability changes of 40!, we did not take into account the fact that if the shock is very positive, then the lender will not return more than what we took.
    ChangeInValueOfLoans1=-np.dot(ChangeInDefaultProbMatrix[ShocksSectorsColumns].to_numpy(), FaceValuesOfLoans['Face Values of Loans'].to_numpy())*(1-RecoveryRate)
    ChangeInValueOfLoans2={'scenario':ChangeInDefaultProbMatrix['scenario'], 'year': ChangeInDefaultProbMatrix['year'],'region': ChangeInDefaultProbMatrix['region'],  'Change in value of loans':ChangeInValueOfLoans1}
    ChangeInValueOfLoans=pd.DataFrame(ChangeInValueOfLoans2)
    ChangeInValueOfLoans['New value of loans']=ChangeInValueOfLoans['Change in value of loans']+FaceValuesOfLoans['Face Values of Loans'].sum()


    LoanValueSimulation=sns.relplot(data=ChangeInValueOfLoans, kind='line', x='year', y='New value of loans', hue='scenario', col='region', height=5, aspect=8/5)
    st.pyplot(LoanValueSimulation)