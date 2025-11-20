EXEMPLARS = [
    {
        "question": "How many Physics laureates were born in the United States?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE toLower(p.category) = 'physics' AND toLower(s.birthCountry) CONTAINS 'united states' RETURN count(DISTINCT s) AS count"
    },
    {
        "question": "Who were the mentors of Marie Curie?",
        "cypher": "MATCH (s:Scholar)-[:MENTORED_BY]->(m:Scholar) WHERE toLower(s.knownName) CONTAINS 'marie curie' RETURN m.knownName"
    },
    {
        "question": "Which scholars won prizes in Physics and were affiliated with University of Cambridge?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:AFFILIATED_WITH]->(i:Institution) WHERE toLower(p.category) = 'physics' AND toLower(i.name) CONTAINS 'cambridge' RETURN s.knownName"
    },
    {
        "question": "List all Chemistry laureates from Germany.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE toLower(p.category) = 'chemistry' AND toLower(s.birthCountry) CONTAINS 'germany' RETURN s.knownName, p.year"
    },
    {
        "question": "How many laureates were born in France?",
        "cypher": "MATCH (s:Scholar) WHERE toLower(s.birthCountry) CONTAINS 'france' RETURN count(DISTINCT s) AS count"
    },
    {
        "question": "Who mentored scholars who later won Nobel prizes?",
        "cypher": "MATCH (m:Scholar)<-[:MENTORED_BY]-(s:Scholar)-[:WON]->(p:Prize) RETURN DISTINCT m.knownName AS mentor"
    },
    {
        "question": "What institutions are affiliated with the most Physics laureates?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:AFFILIATED_WITH]->(i:Institution) WHERE toLower(p.category) = 'physics' RETURN i.name, count(DISTINCT s) AS laureate_count ORDER BY laureate_count DESC LIMIT 10"
    },
    {
        "question": "List Medicine laureates who were born in the 20th century.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE toLower(p.category) = 'medicine' AND s.birthDate >= date('1900-01-01') AND s.birthDate < date('2000-01-01') RETURN s.knownName, s.birthDate"
    },
    {
        "question": "Which scholars won multiple Nobel prizes?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WITH s, count(p) AS prize_count WHERE prize_count > 1 RETURN s.knownName, prize_count"
    },
    {
        "question": "How many Economics prizes were awarded before 1990?",
        "cypher": "MATCH (p:Prize) WHERE toLower(p.category) = 'economics' AND p.year < 1990 RETURN count(p) AS count"
    },
    {
        "question": "Who are the laureates affiliated with MIT?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:AFFILIATED_WITH]->(i:Institution) WHERE toLower(i.name) CONTAINS 'mit' RETURN DISTINCT s.knownName"
    },
    {
        "question": "List all scholars born in Sweden who won prizes.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE toLower(s.birthCountry) CONTAINS 'sweden' RETURN s.knownName, p.category, p.year"
    },
    {
        "question": "What is the average year of Physics prizes awarded?",
        "cypher": "MATCH (p:Prize) WHERE toLower(p.category) = 'physics' RETURN avg(p.year) AS average_year"
    },
    {
        "question": "Which countries have produced the most laureates?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) RETURN s.birthCountry, count(DISTINCT s) AS laureate_count ORDER BY laureate_count DESC LIMIT 10"
    },
    {
        "question": "Who mentored Einstein?",
        "cypher": "MATCH (s:Scholar)-[:MENTORED_BY]->(m:Scholar) WHERE toLower(s.knownName) CONTAINS 'einstein' RETURN m.knownName"
    },
    {
        "question": "List all laureates who were born in the same city as they died.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:BORN_IN]->(bc:City), (s)-[:DIED_IN]->(dc:City) WHERE bc = dc RETURN s.knownName, bc.name"
    },
    {
        "question": "How many female Chemistry laureates are there?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE toLower(p.category) = 'chemistry' AND toLower(s.gender) = 'female' RETURN count(DISTINCT s) AS count"
    },
    {
        "question": "Which laureates were affiliated with both Cambridge and Oxford?",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:AFFILIATED_WITH]->(i1:Institution), (s)-[:AFFILIATED_WITH]->(i2:Institution) WHERE toLower(i1.name) CONTAINS 'cambridge' AND toLower(i2.name) CONTAINS 'oxford' RETURN DISTINCT s.knownName"
    },
    {
        "question": "What is the earliest year a prize was awarded?",
        "cypher": "MATCH (p:Prize) RETURN min(p.year) AS earliest_year"
    },
    {
        "question": "List scholars who won prizes after age 70.",
        "cypher": "MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE p.year - s.birthDate.year > 70 RETURN s.knownName, s.birthDate, p.year, (p.year - s.birthDate.year) AS age_at_award"
    },
]
