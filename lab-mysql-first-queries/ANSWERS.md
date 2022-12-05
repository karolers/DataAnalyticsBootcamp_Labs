**1. Which are the different genres?**
Games
Productivity
Weather
Shopping
Reference
Finance
Music
Utilities
Travel
Social Networking
Sports
Business
Health & Fitness
Entertainment
Photo & Video
Navigation
Education
Lifestyle
Food & Drink
News
Book
Medical
Catalogs

**2. Which is the genre with more apps rated?**
Games

**3. Which is the genre with more apps?**
Games

**4. Which is the one with less?**
Catalogs

**5. Take the 10 apps most rated.**
Facebook
Instagram
Clash of Clans
Temple Run
Pandora - Music & Radio
Pinterest
Bible
Candy Crush Saga
Spotify Music
Angry Birds

**6. Take the 10 apps best rated by users.**
LINE Moments - Capture Your Fun Moments
Cafe Story - Play Cooking & Farming Game
Grumpy Cat's Worst Game Ever
Fitness Sync for Fitbit to Apple Health
Stickman Base Jumper 2
Dungeons of Chaos UNITY EDITION
DELISH KITCHEN - ????????????????????????????????????
????????????IV:????????????
LINE RUSH !
Able Black

**7. Take a look on the data you retrieved in the question 5. Give some insights.**
The 10 apps most rated, were rated more than 800000 thousand times, so we could conclude that these apps will probably be the most used.

**8. Take a look on the data you retrieved in the question 6. Give some insights.**
These apps count of rates start at 1 and the highest count of rate is 10828 for the app PewDiePie: Legend of the Brofist. 

**9. Now compare the data from questions 5 and 6. What do you see?**
The apps that were most rated have slightly lower evaluations from the 10 better rated. Since the first one have a much larger 
sample os user, we could conclude that those ratings have more fundaments.

**10. How could you take the top 3 regarding the user ratings but also the number of votes?**
SELECT track_name, rating_count_tot, user_rating
FROM apple.applestore_windows
ORDER BY user_rating AND rating_count_tot DESC
LIMIT 3;

**11. Do people care about the price? Do some queries, comment why are you doing them and the results you retrieve. What is your conclusion?**
Between the top 10 apps regarding the user ratings but also the number of votes, only two were paid.

The 5 more expensive apps are evaluated between 3.5 and 4.5 and have a rating count between 41 and 927.

The paid apps have a total amount of ratings of over 12MIL and an average rate of 3.73
The free apps have a total amount of ratings of over 79MIL and an average rate of 3.38

We can conclude that the free apps have the most users, consequently more rates. The average rate of the free vs paid apps is similar.