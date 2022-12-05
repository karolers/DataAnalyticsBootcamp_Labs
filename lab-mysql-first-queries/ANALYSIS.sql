SELECT *
FROM apple.applestore_windows;

-- 1
SELECT DISTINCT prime_genre
FROM apple.applestore_windows;

-- 2
SELECT prime_genre, count(*), sum(rating_count_tot)
FROM apple.applestore_windows
GROUP BY prime_genre
ORDER BY sum(rating_count_tot) DESC;

-- 3 & 4
SELECT prime_genre, id, count(*)
FROM apple.applestore_windows
GROUP BY prime_genre
ORDER BY count(*) DESC;

-- 5 & 7 & 9
SELECT track_name, rating_count_tot, user_rating
FROM apple.applestore_windows
ORDER BY rating_count_tot DESC
LIMIT 10;

-- 6 & 8 & 9
SELECT track_name, user_rating, rating_count_tot
FROM apple.applestore_windows
ORDER BY user_rating DESC
LIMIT 10;

-- 10
SELECT track_name, rating_count_tot, user_rating
FROM apple.applestore_windows
ORDER BY user_rating AND rating_count_tot DESC
LIMIT 3;

-- 11
SELECT track_name, rating_count_tot, user_rating, price
FROM apple.applestore_windows
ORDER BY user_rating AND rating_count_tot DESC
LIMIT 10;

SELECT track_name, rating_count_tot, user_rating, price
FROM apple.applestore_windows
ORDER BY price DESC
LIMIT 5;

SELECT sum(rating_count_tot), avg(user_rating)
FROM apple.applestore_windows
WHERE price = 0;

SELECT sum(rating_count_tot), avg(user_rating)
FROM apple.applestore_windows
WHERE price != 0;