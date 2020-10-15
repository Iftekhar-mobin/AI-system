from flask import Flask, render_template, request
import plotly
import json
from plotter import PlotMaker, make_bar_chart, make_pie_chart, make_stack_chart

app = Flask(__name__)


@app.route('/')
def index():
    feature = 'Bar'
    bar = create_plot(feature)
    return render_template('index.html', plot=bar)


def create_plot(feature):
    plotter = PlotMaker('/home/iftekhar/AI-system/visualizer/flask_Dashboard_corpus_visualizer/processed_texts.txt', 1,
                        'podcast 禁止 ios')
    df = plotter.driver()
    word_freq_df = plotter.count_query_parts_frequency()

    if feature == 'Bar':
        data = make_bar_chart(df)
    elif feature == 'Stack_bar':
        data = make_stack_chart(word_freq_df)
    else:
        data = make_pie_chart(df)

    return json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)


@app.route('/bar', methods=['GET', 'POST'])
def change_features():
    feature = request.args['selected']
    graphJSON = create_plot(feature)

    return graphJSON


# @app.route('/query', methods=["GET", "POST"])
# def query():
#     user_query = request.args.get('user_query')
#     print(user_query)
#     return None


if __name__ == '__main__':
    app.run()
