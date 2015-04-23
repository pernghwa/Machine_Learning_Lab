from config import *
import tweepy
import urllib
import urllib2
import base64
import json

class AppAuthHandler(tweepy.auth.AuthHandler):
    TOKEN_URL = 'https://api.twitter.com/oauth2/token'

    def __init__(self, consumer_key, consumer_secret):
        token_credential = urllib.quote(consumer_key) + ':' + urllib.quote(consumer_secret)
        credential = base64.b64encode(token_credential)

        value = {'grant_type': 'client_credentials'}
        data = urllib.urlencode(value)
        req = urllib2.Request(self.TOKEN_URL)
        req.add_header('Authorization', 'Basic ' + credential)
        req.add_header('Content-Type', 'application/x-www-form-urlencoded;charset=UTF-8')

        response = urllib2.urlopen(req, data)
        json_response = json.loads(response.read())
        self._access_token = json_response['access_token']

    def apply_auth(self, url, method, headers, parameters):
        headers['Authorization'] = 'Bearer ' + self._access_token

try:
    auth = tweepy.AppAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    #auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    #auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    twitter_api = tweepy.API(auth)
except Exception:
    print "no connection"
