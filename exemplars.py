EXEMPLARS = [
    {
        "question": "How many Physics laureates are there?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE toLower(p.category) = 'physics' RETURN count(DISTINCT s)"
    },
    {
        "question": "How many scholars won prizes in Chemistry?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE toLower(p.category) = 'chemistry' RETURN count(DISTINCT s)"
    },
    {
        "question": "List all Medicine laureates.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE toLower(p.category) = 'medicine' RETURN s.knownName, p.awardYear"
    },
    {
        "question": "Which scholars won prizes in Physics and were affiliated with University of Cambridge?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:AFFILIATED_WITH]->(i:Institution) WHERE toLower(p.category) = 'physics' AND toLower(i.name) CONTAINS 'cambridge' RETURN s.knownName"
    },
    {
        "question": "Who are the laureates affiliated with MIT?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:AFFILIATED_WITH]->(i:Institution) WHERE toLower(i.name) CONTAINS 'mit' RETURN DISTINCT s.knownName"
    },
    {
        "question": "What institutions are affiliated with the most Physics laureates?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:AFFILIATED_WITH]->(i:Institution) WHERE toLower(p.category) = 'physics' RETURN i.name, count(DISTINCT s) AS laureate_count ORDER BY laureate_count DESC LIMIT 10"
    },
    {
        "question": "Which laureates were affiliated with Harvard University?",
        "cypher": "MATCH (s:Scholar)-[:AFFILIATED_WITH]->(i:Institution) WHERE toLower(i.name) CONTAINS 'harvard' RETURN DISTINCT s.knownName"
    },
    {
        "question": "List laureates born in Paris.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:BORN_IN]->(c:City) WHERE toLower(c.name) CONTAINS 'paris' RETURN s.knownName, p.category"
    },
    {
        "question": "How many laureates were born in cities in the United States?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:BORN_IN]->(city:City)-[:IS_CITY_IN]->(country:Country) WHERE toLower(country.name) CONTAINS 'united states' RETURN count(DISTINCT s)"
    },
    {
        "question": "Which laureates were born in Germany?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:BORN_IN]->(city:City)-[:IS_CITY_IN]->(country:Country) WHERE toLower(country.name) CONTAINS 'germany' RETURN s.knownName, p.category"
    },
    {
        "question": "List scholars who were born and died in the same city.",
        "cypher": "MATCH (s:Scholar)-[:BORN_IN]->(c:City), (s)-[:DIED_IN]->(c) RETURN s.knownName, c.name"
    },
    {
        "question": "How many female Chemistry laureates are there?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE toLower(p.category) = 'chemistry' AND toLower(s.gender) = 'female' RETURN count(DISTINCT s)"
    },
    {
        "question": "List all female laureates.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE toLower(s.gender) = 'female' RETURN s.knownName, p.category, p.awardYear"
    },
    {
        "question": "How many male Physics laureates are there?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE toLower(p.category) = 'physics' AND toLower(s.gender) = 'male' RETURN count(DISTINCT s)"
    },
    {
        "question": "How many Economics prizes were awarded before 1990?",
        "cypher": "MATCH (p:Prize) WHERE toLower(p.category) = 'economics' AND p.awardYear < 1990 RETURN count(p)"
    },
    {
        "question": "What is the earliest year a prize was awarded?",
        "cypher": "MATCH (p:Prize) RETURN min(p.awardYear) AS earliest_year"
    },
    {
        "question": "What is the average year of Physics prizes awarded?",
        "cypher": "MATCH (p:Prize) WHERE toLower(p.category) = 'physics' RETURN avg(p.awardYear) AS average_year"
    },
    {
        "question": "List prizes awarded in 1921.",
        "cypher": "MATCH (p:Prize) WHERE p.awardYear = 1921 RETURN p.category, p.motivation"
    },
    {
        "question": "How many prizes were awarded after 2000?",
        "cypher": "MATCH (p:Prize) WHERE p.awardYear > 2000 RETURN count(p)"
    },
    {
        "question": "Which scholars won multiple Nobel prizes?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WITH s, count(p) AS prize_count WHERE prize_count > 1 RETURN s.knownName, prize_count"
    },
    {
        "question": "How many scholars in total are in the database?",
        "cypher": "MATCH (s:Scholar) RETURN count(s)"
    },
    {
        "question": "List scholars with known names containing Einstein.",
        "cypher": "MATCH (s:Scholar) WHERE toLower(s.knownName) CONTAINS 'einstein' RETURN s.knownName, s.fullName"
    },
    {
        "question": "Which institutions are located in cities in Europe?",
        "cypher": "MATCH (i:Institution)-[:IS_LOCATED_IN]->(city:City)-[:IS_CITY_IN]->(country:Country)-[:IS_COUNTRY_IN]->(continent:Continent) WHERE toLower(continent.name) CONTAINS 'europe' RETURN DISTINCT i.name, city.name, country.name"
    },
    {
        "question": "List laureates affiliated with institutions in the United Kingdom.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:AFFILIATED_WITH]->(i:Institution)-[:IS_LOCATED_IN]->(city:City)-[:IS_CITY_IN]->(country:Country) WHERE toLower(country.name) CONTAINS 'united kingdom' RETURN s.knownName, i.name"
    },
    {
        "question": "What was the motivation for the 1921 Physics prize?",
        "cypher": "MATCH (p:Prize) WHERE toLower(p.category) = 'physics' AND p.awardYear = 1921 RETURN p.motivation"
    },
    {
        "question": "List all prize motivations in Chemistry.",
        "cypher": "MATCH (p:Prize) WHERE toLower(p.category) = 'chemistry' RETURN p.awardYear, p.motivation ORDER BY p.awardYear"
    },
    {
        "question": "Which countries in Europe have laureates?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:BORN_IN]->(city:City)-[:IS_CITY_IN]->(country:Country)-[:IS_COUNTRY_IN]->(continent:Continent) WHERE toLower(continent.name) CONTAINS 'europe' RETURN DISTINCT country.name, count(DISTINCT s) AS laureate_count ORDER BY laureate_count DESC"
    },
    {
        "question": "List scholars born before 1900 who won prizes.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE s.birthDate < '1900-01-01' RETURN s.knownName, s.birthDate, p.category"
    },
    {
        "question": "Which laureates died in London?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:DIED_IN]->(c:City) WHERE toLower(c.name) CONTAINS 'london' RETURN s.knownName, s.deathDate"
    },
    {
        "question": "List laureates who shared their prize.",
        "cypher": "MATCH (s:Scholar)-[w:WON]->(p:Prize) WHERE w.portion IS NOT NULL AND w.portion <> '1' RETURN s.knownName, p.category, p.awardYear, w.portion"
    },
    {
        "question": "Find information about Marie Curie.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE toLower(s.knownName) CONTAINS 'marie curie' RETURN s.knownName, s.fullName, p.category, p.awardYear"
    },
    {
        "question": "Which prizes did Albert Einstein win?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE toLower(s.knownName) CONTAINS 'einstein' RETURN p.category, p.awardYear, p.motivation"
    },
    {
        "question": "How many prizes were awarded in all categories?",
        "cypher": "MATCH (p:Prize) RETURN p.category, count(p) AS prize_count ORDER BY prize_count DESC"
    },
    {
        "question": "List all Medicine laureates from the 21st century.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE toLower(p.category) = 'medicine' AND p.awardYear >= 2000 RETURN s.knownName, p.awardYear ORDER BY p.awardYear"
    },
    {
        "question": "Which category has the most laureates?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) RETURN p.category, count(DISTINCT s) AS laureate_count ORDER BY laureate_count DESC LIMIT 1"
    },
    {
        "question": "Which institutions in the United States have the most laureates?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:AFFILIATED_WITH]->(i:Institution)-[:IS_LOCATED_IN]->(city:City)-[:IS_CITY_IN]->(country:Country) WHERE toLower(country.name) CONTAINS 'united states' RETURN i.name, count(DISTINCT s) AS laureate_count ORDER BY laureate_count DESC LIMIT 10"
    },
    {
        "question": "List all institutions affiliated with Chemistry laureates.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:AFFILIATED_WITH]->(i:Institution) WHERE toLower(p.category) = 'chemistry' RETURN DISTINCT i.name ORDER BY i.name"
    },
    {
        "question": "How many institutions are in the database?",
        "cypher": "MATCH (i:Institution) RETURN count(i)"
    },
    {
        "question": "Which cities have the most institutions?",
        "cypher": "MATCH (i:Institution)-[:IS_LOCATED_IN]->(c:City) RETURN c.name, count(i) AS institution_count ORDER BY institution_count DESC LIMIT 10"
    },
    {
        "question": "How many different cities are birthplaces of laureates?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:BORN_IN]->(c:City) RETURN count(DISTINCT c)"
    },
    {
        "question": "Which city is the birthplace of the most laureates?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:BORN_IN]->(c:City) RETURN c.name, count(DISTINCT s) AS laureate_count ORDER BY laureate_count DESC LIMIT 1"
    },
    {
        "question": "List laureates born in New York.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:BORN_IN]->(c:City) WHERE toLower(c.name) CONTAINS 'new york' RETURN s.knownName, p.category, p.awardYear"
    },
    {
        "question": "How many laureates were born in cities in France?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:BORN_IN]->(city:City)-[:IS_CITY_IN]->(country:Country) WHERE toLower(country.name) CONTAINS 'france' RETURN count(DISTINCT s)"
    },
    {
        "question": "Which city has been the place of death for the most laureates?",
        "cypher": "MATCH (s:Scholar)-[:DIED_IN]->(c:City) RETURN c.name, count(s) AS count ORDER BY count DESC LIMIT 1"
    },
    {
        "question": "List laureates who died in the same country they were born in.",
        "cypher": "MATCH (s:Scholar)-[:BORN_IN]->(bc:City)-[:IS_CITY_IN]->(country:Country), (s)-[:DIED_IN]->(dc:City)-[:IS_CITY_IN]->(country) RETURN s.knownName, country.name"
    },
    {
        "question": "Which year had the most prizes awarded?",
        "cypher": "MATCH (p:Prize) RETURN p.awardYear, count(p) AS prize_count ORDER BY prize_count DESC LIMIT 1"
    },
    {
        "question": "List all laureates from the 1950s.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE p.awardYear >= 1950 AND p.awardYear < 1960 RETURN s.knownName, p.category, p.awardYear ORDER BY p.awardYear"
    },
    {
        "question": "How many prizes were awarded in each decade?",
        "cypher": "MATCH (p:Prize) RETURN (p.awardYear / 10) * 10 AS decade, count(p) AS prize_count ORDER BY decade"
    },
    {
        "question": "What is the most recent prize awarded?",
        "cypher": "MATCH (p:Prize) RETURN max(p.awardYear) AS latest_year"
    },
    {
        "question": "What is the gender distribution of all laureates?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) RETURN s.gender, count(DISTINCT s) AS count ORDER BY count DESC"
    },
    {
        "question": "List female Physics laureates.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE toLower(p.category) = 'physics' AND toLower(s.gender) = 'female' RETURN s.knownName, p.awardYear"
    },
    {
        "question": "Which category has the highest percentage of female laureates?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE toLower(s.gender) = 'female' RETURN p.category, count(DISTINCT s) AS female_count ORDER BY female_count DESC LIMIT 1"
    },
    {
        "question": "What was the prize amount in 1901?",
        "cypher": "MATCH (p:Prize) WHERE p.awardYear = 1901 RETURN p.category, p.prizeAmount"
    },
    {
        "question": "List prizes with the highest adjusted amounts.",
        "cypher": "MATCH (p:Prize) WHERE p.prizeAmountAdjusted IS NOT NULL RETURN p.category, p.awardYear, p.prizeAmountAdjusted ORDER BY p.prizeAmountAdjusted DESC LIMIT 10"
    },
    {
        "question": "What is the average prize amount for Chemistry?",
        "cypher": "MATCH (p:Prize) WHERE toLower(p.category) = 'chemistry' AND p.prizeAmount IS NOT NULL RETURN avg(p.prizeAmount) AS average_amount"
    },
    {
        "question": "List scholars with their full names and known names.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) RETURN s.fullName, s.knownName LIMIT 20"
    },
    {
        "question": "How many laureates have recorded death dates?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE s.deathDate IS NOT NULL RETURN count(DISTINCT s)"
    },
    {
        "question": "List laureates born in the 1930s.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE s.birthDate >= '1930-01-01' AND s.birthDate < '1940-01-01' RETURN s.knownName, s.birthDate, p.category"
    },
    {
        "question": "How many laureates were born in Asia?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:BORN_IN]->(city:City)-[:IS_CITY_IN]->(country:Country)-[:IS_COUNTRY_IN]->(continent:Continent) WHERE toLower(continent.name) CONTAINS 'asia' RETURN count(DISTINCT s)"
    },
    {
        "question": "Which continents have produced Nobel laureates?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:BORN_IN]->(city:City)-[:IS_CITY_IN]->(country:Country)-[:IS_COUNTRY_IN]->(continent:Continent) RETURN DISTINCT continent.name, count(DISTINCT s) AS laureate_count ORDER BY laureate_count DESC"
    },
    {
        "question": "List institutions in North America.",
        "cypher": "MATCH (i:Institution)-[:IS_LOCATED_IN]->(city:City)-[:IS_CITY_IN]->(country:Country)-[:IS_COUNTRY_IN]->(continent:Continent) WHERE toLower(continent.name) CONTAINS 'north america' RETURN DISTINCT i.name, city.name, country.name"
    },
    {
        "question": "Which country has the most Physics laureates?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:BORN_IN]->(city:City)-[:IS_CITY_IN]->(country:Country) WHERE toLower(p.category) = 'physics' RETURN country.name, count(DISTINCT s) AS laureate_count ORDER BY laureate_count DESC LIMIT 1"
    },
    {
        "question": "How many countries have produced at least one laureate?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:BORN_IN]->(city:City)-[:IS_CITY_IN]->(country:Country) RETURN count(DISTINCT country)"
    },
    {
        "question": "List all countries with more than 10 laureates.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:BORN_IN]->(city:City)-[:IS_CITY_IN]->(country:Country) WITH country, count(DISTINCT s) AS laureate_count WHERE laureate_count > 10 RETURN country.name, laureate_count ORDER BY laureate_count DESC"
    },
    {
        "question": "List Chemistry laureates affiliated with European institutions.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:AFFILIATED_WITH]->(i:Institution)-[:IS_LOCATED_IN]->(city:City)-[:IS_CITY_IN]->(country:Country)-[:IS_COUNTRY_IN]->(continent:Continent) WHERE toLower(p.category) = 'chemistry' AND toLower(continent.name) CONTAINS 'europe' RETURN s.knownName, i.name, country.name"
    },
    {
        "question": "Which laureates were born in one country but affiliated with institutions in another?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:BORN_IN]->(bc:City)-[:IS_CITY_IN]->(birthCountry:Country), (s)-[:AFFILIATED_WITH]->(i:Institution)-[:IS_LOCATED_IN]->(ic:City)-[:IS_CITY_IN]->(affCountry:Country) WHERE birthCountry <> affCountry RETURN s.knownName, birthCountry.name AS birth_country, affCountry.name AS affiliation_country LIMIT 20"
    },
    {
        "question": "Find prizes with motivations containing 'discovery'.",
        "cypher": "MATCH (p:Prize) WHERE toLower(p.motivation) CONTAINS 'discovery' RETURN p.category, p.awardYear, p.motivation LIMIT 10"
    },
    {
        "question": "List all prizes with motivations in Physics from 2000 onwards.",
        "cypher": "MATCH (p:Prize) WHERE toLower(p.category) = 'physics' AND p.awardYear >= 2000 RETURN p.awardYear, p.motivation ORDER BY p.awardYear"
    },
    {
        "question": "How many prizes were shared among multiple laureates?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WITH p, count(s) AS laureate_count WHERE laureate_count > 1 RETURN count(p)"
    },
    {
        "question": "Which prize had the most co-winners?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WITH p, count(s) AS laureate_count ORDER BY laureate_count DESC LIMIT 1 RETURN p.category, p.awardYear, laureate_count"
    },
    {
        "question": "List all solo prize winners in Physics.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE toLower(p.category) = 'physics' WITH p, count(s) AS scholar_count, collect(s.knownName) AS names WHERE scholar_count = 1 RETURN names[1] AS laureate, p.awardYear"
    },
    {
        "question": "List all unique prize categories.",
        "cypher": "MATCH (p:Prize) RETURN DISTINCT p.category ORDER BY p.category"
    },
    {
        "question": "How many scholars never won a prize but are in the database?",
        "cypher": "MATCH (s:Scholar) WHERE NOT (s)-[:WON]->(:Prize) RETURN count(s)"
    },
    {
        "question": "Find the oldest laureate by birth date.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE s.birthDate IS NOT NULL RETURN s.knownName, s.birthDate ORDER BY s.birthDate ASC LIMIT 1"
    },
    {
        "question": "Find laureates who won prizes in their birth decade.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE s.birthDate IS NOT NULL AND substring(s.birthDate, 0, 3) = substring(string(p.awardYear), 0, 3) RETURN s.knownName, s.birthDate, p.awardYear LIMIT 10"
    },
]
