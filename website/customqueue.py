import sqlite3

root_dir = "/home/lamparter/stableDiffusion"

class CustomQueue:
    def __init__(self):
        self.conn = sqlite3.connect('{0}/queue.db'.format(root_dir))
        #self.conn.execute("DROP TABLE QUEUE;")
        self.conn.execute('''CREATE TABLE IF NOT EXISTS QUEUE
         (ID INT PRIMARY KEY     NOT NULL,
         PROMPT           TEXT    NOT NULL,
         NAME TEXT NOT NULL,
         USERID TEXT NOT NULL,
         STEPS            TEXT     NOT NULL,
         FILEPATH        TEXT,
         GLOBAL         TEXT,
         USER TEXT,
         FINISH INT);''')
        self.conn.commit()
        
    
    def enqueue(self, prompt, model_name, userid, image_save_path_global, image_save_path_user, num_inference_steps, filepath, wait=True):
        ID = [row[0] for row in self.conn.execute("SELECT MAX(ID) FROM QUEUE;")] 
        if ID[0] == None:
            ID = 0
        if isinstance(ID, list):  
            ID = ID[0]
        insertstring = "INSERT INTO QUEUE (ID,PROMPT,NAME,USERID,STEPS,FILEPATH,GLOBAL,USER,FINISH) VALUES ({0}, '{1}', '{2}', '{3}', '{4}', '{5}', '{6}', '{7}', {8})".format(
            ID+1,
            prompt,
            model_name,
            userid,
            num_inference_steps,
            filepath,
            image_save_path_global,
            image_save_path_user,
            0
        )
        print(insertstring)
        self.conn.execute(insertstring)
        self.conn.commit()
        while wait == True:
            row = [element for element in self.conn.execute("SELECT * FROM QUEUE where ID = {0};".format(ID+1))]
            print(row[-1])
            if row[0][-1] == 1:
                wait = False
        self.conn.execute("DELETE from QUEUE where ID = {0};".format(ID+1))
        self.conn.commit()
        print("\n\n FINISH \n\n")
    
    def dequeue(self):
        row = [entry for entry in self.conn.execute("SELECT * FROM QUEUE ORDER BY ID DESC LIMIT 1;")]
        return row
    
    def finish(self, ID):
        print("UPDATE QUEUE set FINISH = 1 where ID = {0};".format(ID))
        self.conn.execute("UPDATE QUEUE set FINISH = 1 where ID = {0};".format(ID))
        self.conn.commit()

        

