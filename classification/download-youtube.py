import youtube_dl

ydl_opts = {}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    url = 'https://www.youtube.com/watch?v=8w8bzb-BWw0'
    ydl.download([url])