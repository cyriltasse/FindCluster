import pymysql.cursors


def test():
    # Connect to the database
    connection = pymysql.connect(host='gavo.mpa-garching.mpg.de',
                                 #user='user',
                                 #password='passwd',
                                 db='Millennium',
                                 #charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)

    try:
        # with connection.cursor() as cursor:
        #     # Create a new record
        #     sql = "INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)"
        #     cursor.execute(sql, ('webmaster@python.org', 'very-secret'))
            
        #     # connection is not autocommit by default. So you must commit to save
        #     # your changes.
        #     connection.commit()

        with connection.cursor() as cursor:
            # Read a single record
            sql = "select D.galaxyID, D.x, D.y, D.z, D.redshift, D.snapnum, D.stellarMass from millimil..DeLucia2006a D where snapnum=63 and x between 0 and 100 and y between 0 and 100 and z between 0 and 100"
            cursor.execute(sql, ('webmaster@python.org',))
            result = cursor.fetchone()
            print(result)
    finally:
        connection.close()
            
