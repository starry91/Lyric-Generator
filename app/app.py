import flask
import runner as rc
# import five_words as five

load_path = "../save/models/lyricmodel/PinkFloyd.ckpt-70000"
artist_name = "PinkFloyd"
test = True

def infer(prime_text):
    prime_text = prime_text.lower()
    lyricmodel = rc.LyricGenRunner(load_path, artist_name, test, prime_text)
    result = lyricmodel.test(prime_text)
    return result

app = flask.Flask(__name__, template_folder="templates")

@app.route('/', methods=['GET'])
def webpage():
    return flask.render_template('index.html')

# @app.route('/five', methods=['GET'])
# def webpage_five():
#     return flask.render_template('index_two.html')

@app.route('/lyrics', methods = ['POST'])
def search():
    content = flask.request.get_json(silent = True)
    input_text = content['search_text']
    generated = infer(input_text)
    return flask.jsonify({'generated': generated})

# @app.route('/lyricsfive', methods = ['POST'])
# def searchfive():
#     content = flask.request.get_json(silent = True)
#     input_text = content['search_text']
#     input_text = ' '.join(five.get_sentences(input_text))
#     try:
#         input_text = input_text.lower()
#         generated = infer(input_text)
#     except:
#         generated = input_text
#     return flask.jsonify({'generated': generated})

@app.route('/test', methods=['GET'])
def test():
    return flask.jsonify({'ping': 'ping_data'})

app.config['TEMPLATES_AUTO_RELOAD'] = True
app.run('0.0.0.0', 8008, debug = True)