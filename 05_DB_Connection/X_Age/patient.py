import cx_Oracle as ocl


oracle_dsn = ocl.makedsn(host="localhost", port=1521, sid="xe")
conn = ocl.connect(dsn=oracle_dsn, user="hr", password="hr")


def get_all_list():
    sql = "SELECT * FROM x_ray"
    cursor = conn.cursor()
    cursor.execute(sql)
    patient_list = cursor.fetchall()
    return patient_list


def get_list_by_id(patient_id):
    # sql = "SELECT * FROM x_ray WHERE patient_id=:patient_id"
    sql = "SELECT * FROM x_ray WHERE patient_id LIKE :patient_id"
    cursor = conn.cursor()
    cursor.execute(sql, ('%'+patient_id+'%',))
    patient_list = cursor.fetchall()
    return patient_list

def update_bone_age(bone_age, patient_id, examination_number):
    sql = "UPDATE x_ray SET bone_age=:bone_age WHERE patient_id=:patient_id and examination_number=:examination_number"
    cursor = conn.cursor()
    cursor.execute(sql, (bone_age, patient_id, examination_number))
    # conn.commit()

