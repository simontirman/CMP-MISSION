#import pandas as pd
#import streamlit as st
#import plotly.express as px
#from IPython.display import Image, HTML
#from IPython.display import display, HTML
#from ipywidgets import interact, interactive, fixed, interact_manual
#import ipywidgets as widgets
#from ipywidgets import *
#import pandas as pd
#from PIL import Image
#from io import BytesIO
#pd.set_option('display.max_colwidth', -1)
#import warnings
#warnings.filterwarnings('ignore')
#import string
#
#@st.cache
#def get_data():
#    data = pd.read_csv("scrap_data.csv")
#    return data.iloc[:,1:]
#
#
#            
#            
#df = get_data()
#
#st.title("Prototype CMP ")
#st.markdown("Résultats préliminaires pour l'arborescence donné")
#st.header("Exploration du Data")
#
#st.markdown("Les premières ligne des données récupérées sur Alibaba, Amazon, Walmart, etc..")
##st.dataframe(df.head(8))
#
#defaultcols = ["Title", "Source", "Image" ,"Search Keywords", "Link"]
#cols = st.multiselect("Columns", df.columns.tolist(), default=defaultcols)
#st.dataframe(df[cols].head(35))
#
#CATEGORIES = df["Search Keywords"].unique()
#CATEGORIES_SELECTED = st.multiselect('Dans quelle(e) catégorie(s) souhaites-tu connaître les best-sellers du moment?', CATEGORIES)
#
#mask_categories = df["Search Keywords"].isin(CATEGORIES_SELECTED)
#
#data = df[mask_categories]
#data.reset_index(inplace=True)
#data = data.iloc[:,1:]
#st.dataframe(data.head(20))
#
#import urllib.request
#for im in range(len(data.Link)):
#    #pic ='"%s"'%data.loc[im,'Image']
#    urllib.request.urlretrieve(data.loc[im,'Image'], "local-filename.jpg")
#    st.image("local-filename.jpg",width=300, use_column_width=False, caption=data.loc[im,'Title'])
#    #st.image(pic)
##    
#
#pic = st.selectbox("Picture choices", list(pics.keys()), 0)
#st.image(pics[pic], use_column_width=True, caption=pics[pic])

import io
from typing import List, Optional

import markdown
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly import express as px
from plotly.subplots import make_subplots
import io
from typing import List, Optional
import markdown
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from IPython.display import Image, HTML
from IPython.display import display, HTML
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from ipywidgets import *
import pandas as pd
from PIL import Image
from io import BytesIO
pd.set_option('display.max_colwidth', -1)
import warnings
warnings.filterwarnings('ignore')
import string
COLOR='black'
def get_data():
    #data = pd.read_csv("/Users/simontirman/Downloads/df_filtered11.csv")
    data = pd.read_csv("df_filtered33.csv")
    
    return data

@st.cache
def get_dataframe() -> pd.DataFrame():
    """Dummy DataFrame"""
    data = [
        {"quantity": 1, "price": 2},
        {"quantity": 3, "price": 5},
        {"quantity": 4, "price": 8},
    ]
    return pd.DataFrame(data)






class Cell:
    """A Cell can hold text, markdown, plots etc."""

    def __init__(
        self,
        class_: str = None,
        grid_column_start: Optional[int] = None,
        grid_column_end: Optional[int] = None,
        grid_row_start: Optional[int] = None,
        grid_row_end: Optional[int] = None,
    ):
        self.class_ = class_
        self.grid_column_start = grid_column_start
        self.grid_column_end = grid_column_end
        self.grid_row_start = grid_row_start
        self.grid_row_end = grid_row_end
        self.inner_html = ""

    def _to_style(self) -> str:
        return f"""
.{self.class_} {{
    grid-column-start: {self.grid_column_start};
    grid-column-end: {self.grid_column_end};
    grid-row-start: {self.grid_row_start};
    grid-row-end: {self.grid_row_end};
}}
"""

    def text(self, text: str = ""):
        self.inner_html = text

    def markdown(self, text):
        self.inner_html = markdown.markdown(text)

    def dataframe(self, dataframe: pd.DataFrame):
        self.inner_html = dataframe.to_html()

    def plotly_chart(self, fig):
        self.inner_html = f"""
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<body>
    <p>This should have been a plotly plot.
    But since *script* tags are removed when inserting MarkDown/ HTML i cannot get it to workto work.
    But I could potentially save to svg and insert that.</p>
    <div id='divPlotly'></div>
    <script>
        var plotly_data = {fig.to_json()}
        Plotly.react('divPlotly', plotly_data.data, plotly_data.layout);
    </script>
</body>
"""

    def pyplot(self, fig=None, **kwargs):
        string_io = io.StringIO()
        plt.savefig(string_io, format="svg", fig=(2, 2))
        svg = string_io.getvalue()[215:]
        plt.close(fig)
        self.inner_html = '<div height="200px">' + svg + "</div>"

    def _to_html(self):
        return f"""<div class="box {self.class_}">{self.inner_html}</div>"""


class Grid:
    """A (CSS) Grid"""

    def __init__(
        self,
        template_columns="1 1 1",
        gap="10px",
        background_color='black',
        color='black',
    ):
        self.template_columns = template_columns
        self.gap = gap
        self.background_color = background_color
        self.color = color
        self.cells: List[Cell] = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        st.markdown(self._get_grid_style(), unsafe_allow_html=True)
        st.markdown(self._get_cells_style(), unsafe_allow_html=True)
        st.markdown(self._get_cells_html(), unsafe_allow_html=True)

    def _get_grid_style(self):
        return f"""
<style>
    .wrapper {{
    display: grid;
    grid-template-columns: {self.template_columns};
    grid-gap: {self.gap};
    background-color: {self.background_color};
    color: {self.color};
    }}
    .box {{
    background-color: {self.color};
    color: {self.background_color};
    border-radius: 5px;
    padding: 20px;
    font-size: 150%;
    }}
    table {{
        color: {self.color}
    }}
</style>
"""

    def _get_cells_style(self):
        return (
            "<style>"
            + "\n".join([cell._to_style() for cell in self.cells])
            + "</style>"
        )

    def _get_cells_html(self):
        return (
            '<div class="wrapper">'
            + "\n".join([cell._to_html() for cell in self.cells])
            + "</div>"
        )

    def cell(
        self,
        class_: str = None,
        grid_column_start: Optional[int] = None,
        grid_column_end: Optional[int] = None,
        grid_row_start: Optional[int] = None,
        grid_row_end: Optional[int] = None,
    ):
        cell = Cell(
            class_=class_,
            grid_column_start=grid_column_start,
            grid_column_end=grid_column_end,
            grid_row_start=grid_row_start,
            grid_row_end=grid_row_end,
        )
        self.cells.append(cell)
        return cell


def select_block_container_style():
    """Add selection section for setting setting the max-width and padding
    of the main block container"""
    st.sidebar.header("Configuration de la Page de Recommendation")
    max_width_100_percent = st.sidebar.checkbox("Taille d'image maximale (500px) ?", False)
    if not max_width_100_percent:
        max_width = st.sidebar.slider("Choisissez la taille de l'image (en px). ", 50, 500, 300, 50)
    else:
        max_width = 500
    
    dark_theme = st.sidebar.checkbox("Dark Theme?", False)
    if dark_theme:
        st.markdown("""
<style>
body {
    color: #fff;
    background-color: #111;
    etc. 
}
</style>
    """, unsafe_allow_html=True)

    type_cat = st.sidebar.radio('Quel type de recommendation voulez-vous consulter ?',('Arborescence', 'Classement Produits'), key='hello')

    return max_width, type_cat

  


def _set_block_container_style(
    max_width: int = 1200,
    max_width_100_percent: bool = False,
    padding_top: int = 5,
    padding_right: int = 1,
    padding_left: int = 1,
    padding_bottom: int = 10,
):
    if max_width_100_percent:
        max_width_str = f"max-width: 100%;"
    else:
        max_width_str = f"max-width: {max_width}px;"
    st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        {max_width_str}
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
    .reportview-container .main {{
        color: {'black'};
        background-color: {'black'};
    }}
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache
def get_dataframe() -> pd.DataFrame():
    """Dummy DataFrame"""
    data = [
        {"quantity": 1, "price": 2},
        {"quantity": 3, "price": 5},
        {"quantity": 4, "price": 8},
    ]
    return pd.DataFrame(data)


def get_plotly_fig():
    """Dummy Plotly Plot"""
    return px.line(data_frame=get_dataframe(), x="quantity", y="price")


def get_matplotlib_plt():
    get_dataframe().plot(kind="line", x="quantity", y="price", figsize=(5, 3))


def get_plotly_subplots():
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Table 4"),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "table"}],
        ],
    )
    ix=0
    urllib.request.urlretrieve(data.loc[ix,'Link'], "local-filename.jpg")
    
    #fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]), row=1, col=1)
    #fig.add_trace(st.image("local-filename.jpg",width=250, use_column_width=False, caption= str(data.loc[0,'Cluster']) + ' \n Importance Score ' + str(data.loc[0,'Count'])))

    fig.add_trace(go.Scatter(x=[20, 30, 40], y=[50, 60, 70]), row=1, col=2)

    fig.add_trace(go.Scatter(x=[300, 400, 500], y=[600, 700, 800]), row=2, col=1)

    fig.add_table(
        header=dict(values=["A Scores", "B Scores"]),
        cells=dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]]),
        row=2,
        col=2,
    )

    if COLOR == "black":
        template="plotly"
    else:
        template ="plotly_dark"
    fig.update_layout(
        height=500,
        width=700,
        title_text="Plotly Multiple Subplots with Titles",
        template=template,
    )
    return fig 


def paginator(label, items, items_per_page=10, on_sidebar=True):
    """Lets the user paginate a set of items.
    Parameters
    ----------
    label : str
        The label to display over the pagination widget.
    items : Iterator[Any]
        The items to display in the paginator.
    items_per_page: int
        The number of items to display per page.
    on_sidebar: bool
        Whether to display the paginator widget on the sidebar.
        
    Returns
    -------
    Iterator[Tuple[int, Any]]
        An iterator over *only the items on that page*, including
        the item's index.
    Example
    -------
    This shows how to display a few pages of fruit.
    >>> fruit_list = [
    ...     'Kiwifruit', 'Honeydew', 'Cherry', 'Honeyberry', 'Pear',
    ...     'Apple', 'Nectarine', 'Soursop', 'Pineapple', 'Satsuma',
    ...     'Fig', 'Huckleberry', 'Coconut', 'Plantain', 'Jujube',
    ...     'Guava', 'Clementine', 'Grape', 'Tayberry', 'Salak',
    ...     'Raspberry', 'Loquat', 'Nance', 'Peach', 'Akee'
    ... ]
    ...
    ... for i, fruit in paginator("Select a fruit page", fruit_list):
    ...     st.write('%s. **%s**' % (i, fruit))
    """

    # Figure out where to display the paginator
    if on_sidebar:
        location = st.sidebar.empty()
    else:
        location = st.empty()

    # Display a pagination selectbox in the specified location.
    items = list(items)
    n_pages = len(items)
    n_pages = (len(items) - 1) // items_per_page + 1
    page_format_func = lambda i: "Page %s" % i
    page_number = location.selectbox(label, range(n_pages), format_func=page_format_func)

    # Iterate over the items in the page to let the user display them.
    min_index = page_number * items_per_page
    max_index = min_index + items_per_page
    import itertools
    return itertools.islice(enumerate(items), min_index, max_index)

selections = select_block_container_style()
max_width = selections[0]
type_cat = selections[1]


categories = pd.read_csv('families.csv', delimiter=';')
categories.columns=['a','b']

fam_dict = dict(zip(categories.a, categories.b))



df = get_data()
#df['Category'] = df.replace({"Category": fam_dict})['Category']
logo = Image.open('/Users/simontirman/Downloads/logo_Cmp_Paris.png')
st.image(logo,width=max_width, use_column_width=True )
st.title("Recherche des Produits Tendances Selon l'Arborescence CMP")
st.markdown('<style>h1{color: grey;}</style>', unsafe_allow_html=True)

st.header("Produits Recommandés : ")


defaultcols = ["Product", "Category", "Link" ,"Cluster", "Original", "Count"]
#cols = st.multiselect("Columns", df.columns.tolist(), default=defaultcols)
#st.dataframe(df[cols].head(35))

CATEGORIES = df["Category"].unique()

#st.dataframe(data.head(20))
#data_test=data
#sunset_imgs = data_test['Link']
#capt =  data_test['Cluster']
#image_iterator = paginator("Select a sunset page", sunset_imgs)
#st.text(image_iterator)
#indices_on_page, images_on_page = map(list, zip(*image_iterator))
#st.image(images_on_page,width=250, use_column_width=False, caption=capt)

import urllib.request
count = 0

st.markdown('Cette sélection de produit est constituée par des algorithmes d\'intelligence artificielle, prenant en compte de multiples variables.')
st.markdown('Par conséquent, il est important de considérer à la fois l\'image et la description produit pour déterminer la nature réelle du produit. ')


if (type_cat=='Arborescence'):
    CATEGORIES_SELECTED = st.multiselect('Dans quelle(e) catégorie(s) souhaites-tu connaître les best-seller du moment?', CATEGORIES)

    mask_categories = df["Category"].isin(CATEGORIES_SELECTED)
    data = df[mask_categories]
    data.reset_index(inplace=True)
    data = data.iloc[:,1:]
    for im in range(len(data.Link)):
        try:
            #pic ='"%s"'%data.loc[im,'Image']
            #urllib.request.urlretrieve(data.loc[im,'Link'], "local-filename.jpg")
            response = requests.get(data.loc[im,'Link'])
            img = Image.open(BytesIO(response.content)).convert('RGB')
            #img = Image.open(urllib.request.urlretrieve(data.loc[im,'Link'], "local-filename.jpg")).convert('RGB').save('new.jpeg')
            st.title(str('Ranking : ') + str(count+1))
            st.markdown('<style>h1{color: grey;}</style>', unsafe_allow_html=True)
            st.image(img,width=max_width, use_column_width=False)#, caption= str('Product : ' + data.loc[im,'Cluster']) + ' \n Produit : ' + str(data.loc[im,'Cluster FR'])+ ' \n Importance Score ' + str(data.loc[im,'Count']))
            
            #st.image("local-filename.jpg",width=max_width, use_column_width=False)#, caption= str('Product : ' + data.loc[im,'Cluster']) + ' \n Produit : ' + str(data.loc[im,'Cluster FR'])+ ' \n Importance Score ' + str(data.loc[im,'Count']))
            st.markdown( str('Product (EN) : ' + data.loc[im,'Cluster']))
            st.markdown(str('Produit (FR) : ' + data.loc[im,'Cluster FR']))
            st.markdown(str('Importance Score : ' + str(data.loc[im,'Count'])))
            count+=1
        except:
            #st.text(data.loc[im,'Link'])
            pass

else :
    st.title('Classement Produits')




#st.sidebar.title("Layout and Style Experiments")
#st.sidebar.header("Settings")







##    #st.image(pic)
##

#
#data['img_html'] = data['Image']\
#    .str.replace(
#        '(.*)', 
#        '<img src="\\1" style="max-height:124px;"></img>'
#    )
#with pd.option_context('display.max_colwidth', 10000):
#    HTML(data.to_html(escape=False))
#    
#    
#def get_thumbnail(path):
#    i = Image.open(path)
#    i.thumbnail((150, 150), Image.LANCZOS)
#    return i
#
#def image_base64(im):
#    if isinstance(im, str):
#        im = get_thumbnail(im)
#    with BytesIO() as buffer:
#        im.save(buffer, 'jpeg')
#        return base64.b64encode(buffer.getvalue()).decode()
#
#def image_formatter(im):
#    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'
#
#def path_to_image_html(path):
#    return '<img src="'+ path + '" style=max-height:124px;"/>'
#
#HTML(data[['Title', 'img_html']].to_html(escape=False ,formatters=dict(column_name_with_image_links=path_to_image_html)))
#    
#
#
#def init_groupings(df):
# categorical_columns = df.nunique()[df.nunique() < 50].index
# grouping = st.sidebar.selectbox('Grouper par :', categorical_columns)
# remaining_options = [' — '] + list( set(categorical_columns)-set([grouping]) )
# grouping2 = st.sidebar.selectbox('et par :', remaining_options)
# final_group_keys = [grouping, grouping2] if grouping2 != ' — ' else [grouping]
# return final_group_keys
#
#
#def init_SelectionsLabels(df, grouping_keys):
#     for element in grouping_keys:
#        level_options = list(pd.value_counts(df[element]).index)
#        globals()['SelectedLabelsFor' + element] = st.sidebar.multiselect(label='Choice of levels for'+ element, options=level_options)
#                
#def displayMe(df, grouping_keys):
#    import sys
#    thismodule = sys.modules[__name__]
#    for element in grouping_keys:
#        st.write(globals()['SelectedLabelsFor'+element])
#        df_mini = df[ df[element].isin(getattr(thismodule, 'SelectedLabelsFor'+element)) ]
#        st.dataframe(df_mini)
#        
#def path_to_image_html(path):
#    '''
#     This function essentially convert the image url to 
#     '<img src="'+ path + '"/>' format. And one can put any
#     formatting adjustments to control the height, aspect ratio, size etc.
#     within as in the below example. 
#    '''
#
#    return '<img src="'+ path + '" style=max-height:124px;"/>'
#
#
#        
#final_group_keys = init_groupings(df) 
#init_SelectionsLabels(df, final_group_keys)
#displayMe(df,final_group_keys)




        
## Create a plot to display the Map.
 
 ## important: to display in the Streamlit app
#st.header("Which host has the most properties listed?")
#listingcounts = df.host_id.value_counts()
#top_host_1 = df.query('host_id==@listingcounts.index[0]')
#top_host_2 = df.query('host_id==@listingcounts.index[1]')
#st.write(f"""**{top_host_1.iloc[0].host_name}** is at the top with {listingcounts.iloc[0]} property listings.
#**{top_host_2.iloc[1].host_name}** is second with {listingcounts.iloc[1]} listings. Following are randomly chosen
#listings from the two displayed as JSON using [`st.json`](https://streamlit.io/docs/api.html#streamlit.json).""")
#
#st.json({top_host_1.iloc[0].host_name: top_host_1\
#    [["name", "neighbourhood", "room_type", "minimum_nights", "price"]]\
#        .sample(2, random_state=4).to_dict(orient="records"),
#        top_host_2.iloc[0].host_name: top_host_2\
#    [["name", "neighbourhood", "room_type", "minimum_nights", "price"]]\
#        .sample(2, random_state=4).to_dict(orient="records")})
#
#st.header("What is the distribution of property price?")
#st.write("""Select a custom price range from the side bar to update the histogram below displayed as a Plotly chart using
#[`st.plotly_chart`](https://streamlit.io/docs/api.html#streamlit.plotly_chart).""")
#values = st.sidebar.slider("Price range", float(df.price.min()), float(df.price.clip(upper=1000.).max()), (50., 300.))
#f = px.histogram(df.query(f"price.between{values}"), x="price", nbins=15, title="Price distribution")
#f.update_xaxes(title="Price")
#f.update_yaxes(title="No. of listings")
#st.plotly_chart(f)
#
#st.header("What is the distribution of availability in various neighborhoods?")
#st.write("Using a radio button restricts selection to only one option at a time.")
#st.write("💡 Notice how we use a static table below instead of a data frame. \
#Unlike a data frame, if content overflows out of the section margin, \
#a static table does not automatically hide it inside a scrollable area. \
#Instead, the overflowing content remains visible.")
#neighborhood = st.radio("Neighborhood", df.neighbourhood_group.unique())
#show_exp = st.checkbox("Include expensive listings")
#show_exp = " and price<200" if not show_exp else ""
#
#@st.cache
#def get_availability(show_exp, neighborhood):
#    return df.query(f"""neighbourhood_group==@neighborhood{show_exp}\
#        and availability_365>0""").availability_365.describe(\
#            percentiles=[.1, .25, .5, .75, .9, .99]).to_frame().T
#
#st.table(get_availability(show_exp, neighborhood))
#st.write("At 169 days, Brooklyn has the lowest average availability. At 226, Staten Island has the highest average availability.\
#    If we include expensive listings (price>=$200), the numbers are 171 and 230 respectively.")
#st.markdown("_**Note:** There are 18431 records with `availability_365` 0 (zero), which I've ignored._")
#
#df.query("availability_365>0").groupby("neighbourhood_group")\
#    .availability_365.mean().plot.bar(rot=0).set(title="Average availability by neighborhood group",
#        xlabel="Neighborhood group", ylabel="Avg. availability (in no. of days)")
#st.pyplot()
#
#st.header("Properties by number of reviews")
#st.write("Enter a range of numbers in the sidebar to view properties whose review count falls in that range.")
#minimum = st.sidebar.number_input("Minimum", min_value=0)
#maximum = st.sidebar.number_input("Maximum", min_value=0, value=5)
#if minimum > maximum:
#    st.error("Please enter a valid range")
#else:
#    df.query("@minimum<=number_of_reviews<=@maximum").sort_values("number_of_reviews", ascending=False)\
#        .head(50)[["name", "number_of_reviews", "neighbourhood", "host_name", "room_type", "price"]]
#
#st.write("486 is the highest number of reviews and two properties have it. Both are in the East Elmhurst \
#    neighborhood and are private rooms with prices $65 and $45. \
#    In general, listings with >400 reviews are priced below $100. \
#    A few are between $100 and $200, and only one is priced above $200.")
#st.header("Images")
#pics = {
#    "Cat": "https://cdn.pixabay.com/photo/2016/09/24/22/20/cat-1692702_960_720.jpg",
#    "Puppy": "https://cdn.pixabay.com/photo/2019/03/15/19/19/puppy-4057786_960_720.jpg",
#    "Sci-fi city": "https://storage.needpix.com/rsynced_images/science-fiction-2971848_1280.jpg"
#}
#pic = st.selectbox("Picture choices", list(pics.keys()), 0)
#st.image(pics[pic], use_column_width=True, caption=pics[pic])
#
#st.markdown("## Party time!")
#st.write("Yay! You're done with this tutorial of Streamlit. Click below to celebrate.")
#btn = st.button("Celebrate!")
#if btn:
#    st.balloons()
#