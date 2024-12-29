from flask import Flask, request, jsonify, render_template, redirect
import sqlite3

app = Flask(__name__)

def init_db():
    with sqlite3.connect("parking.db") as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS VehicleType (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT NOT NULL
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS PriceConvention (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            VehicleTypeID INTEGER NOT NULL,
            Time TEXT NOT NULL,
            Price REAL NOT NULL,
            TicketType TEXT NOT NULL,
            StartTime TEXT NOT NULL,
            EndTime TEXT NOT NULL,
            FOREIGN KEY (VehicleTypeID) REFERENCES VehicleType(ID)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ActualParkingFee (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            VehicleTypeID INTEGER NOT NULL,
            PlateNumber TEXT NOT NULL,
            EntryTime TEXT NOT NULL,
            ExitTime TEXT,
            Status TEXT NOT NULL,
            TotalFee REAL,
            Note TEXT,
            FOREIGN KEY (VehicleTypeID) REFERENCES VehicleType(ID)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Account (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Username TEXT NOT NULL,
            Password TEXT NOT NULL
        )
        ''')



# API POSTMAN POST price_convention
# @app.route('/price_convention', methods=['POST'])
# def add_price_convention():
#     data = request.get_json()
#     vehicle_type_id = data.get('VehicleTypeID')
#     time = data.get('Time')
#     price = data.get('Price')
#     ticket_type = data.get('TicketType')
#     start_time = data.get('StartTime')
#     end_time = data.get('EndTime')
#     with sqlite3.connect("parking.db") as conn:
#         cursor = conn.cursor()
#         cursor.execute('''
#         INSERT INTO PriceConvention (VehicleTypeID, Time, Price, TicketType, StartTime, EndTime)
#         VALUES (?, ?, ?, ?, ?, ?)
#         ''', (vehicle_type_id, time, price, ticket_type, start_time, end_time))
#         conn.commit()
#         return jsonify({"message": "Price convention added successfully."}), 201

@app.route('/price_convention', methods=['POST'])
def add_price_convention():
    try:
        # Lấy dữ liệu từ form
        vehicle_type_id = request.form.get('VehicleTypeID')
        time = request.form.get('Time')
        price = request.form.get('Price')
        ticket_type = request.form.get('TicketType')
        start_time = request.form.get('StartTime')
        end_time = request.form.get('EndTime')

        # Thêm vào database
        with sqlite3.connect("parking.db") as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO PriceConvention (VehicleTypeID, Time, Price, TicketType, StartTime, EndTime)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (vehicle_type_id, time, price, ticket_type, start_time, end_time))
            conn.commit()

        return redirect('/price_convention')  # Chuyển hướng về trang chính
    except Exception as e:
        return render_template('price_convention.html', error=str(e))


# API POSTMAN PUT price_convention
# @app.route('/price_convention/<int:id>', methods=['PUT'])
# def update_price_convention(id):
#     try:
#         data = request.json
#         with sqlite3.connect("parking.db") as conn:
#             cursor = conn.cursor()
#             cursor.execute('''
#                 UPDATE PriceConvention
#                 SET VehicleTypeID = ?, Time = ?, Price = ?, TicketType = ?, StartTime = ?, EndTime = ?
#                 WHERE ID = ?
#             ''', (data['VehicleTypeID'], data['Time'], data['Price'], data['TicketType'], data['StartTime'], data['EndTime'], id))
#             conn.commit()
#         return jsonify({"message": "PriceConvention updated successfully!"}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# API PUT CỦa price_convention trên HTML
@app.route('/price_convention/update/<int:id>', methods=['POST'])
def update_price_convention(id):
    try:
        # Lấy dữ liệu từ form
        vehicle_type_id = request.form.get('VehicleTypeID')
        time = request.form.get('Time')
        price = request.form.get('Price')
        ticket_type = request.form.get('TicketType')
        start_time = request.form.get('StartTime')
        end_time = request.form.get('EndTime')

        # Cập nhật dữ liệu trong database
        with sqlite3.connect("parking.db") as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE PriceConvention
                SET VehicleTypeID = ?, Time = ?, Price = ?, TicketType = ?, StartTime = ?, EndTime = ?
                WHERE ID = ?
            ''', (vehicle_type_id, time, price, ticket_type, start_time, end_time, id))
            conn.commit()

        return redirect('/price_convention')  # Chuyển hướng về trang chính
    except Exception as e:
        return render_template('price_convention.html', error=str(e))


# API POSTMAN của DELETE price_convention
# @app.route('/price_convention/<int:id>', methods=['DELETE'])
# def delete_price_convention(id):
#     try:
#         with sqlite3.connect("parking.db") as conn:
#             cursor = conn.cursor()
#             cursor.execute('DELETE FROM PriceConvention WHERE ID = ?', (id,))
#             conn.commit()
#         return jsonify({"message": "PriceConvention deleted successfully!"}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

@app.route('/price_convention/delete/<int:id>', methods=['POST'])
def delete_price_convention(id):
    try:
        # Xóa dữ liệu trong database
        with sqlite3.connect("parking.db") as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM PriceConvention WHERE ID = ?', (id,))
            conn.commit()

        # Sau khi xóa, lấy lại dữ liệu để render giao diện
        return redirect('/price_convention')  # Chuyển hướng lại trang Price Convention
    except Exception as e:
        # Lấy lại dữ liệu để tránh lỗi giao diện
        with sqlite3.connect("parking.db") as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM PriceConvention')
            rows = cursor.fetchall()
            data = [dict(row) for row in rows]
            total_pages = 1  # Hoặc tính toán lại số trang nếu cần
        return render_template(
            'price_convention.html',
            price_conventions=data,
            current_page=1,
            total_pages=total_pages,
            error=str(e)
        )


# API GET price_convention trên POSTMAN
# @app.route('/price_convention', methods=['GET'])
# def get_all_price_conventions():
#     try:
#         with sqlite3.connect("parking.db") as conn:
#             conn.row_factory = sqlite3.Row
#             cursor = conn.cursor()
#             cursor.execute('SELECT * FROM PriceConvention')
#             rows = cursor.fetchall()
#             data = [dict(row) for row in rows]
#         return jsonify(data), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# Render ra HTML của API GET price_convention
# @app.route('/price_convention', methods=['GET'])
# def get_all_price_conventions():
#     try:
#         # Lấy trang hiện tại từ query string
#         page = request.args.get('page', 1, type=int)
#         per_page = 5  # Số lượng bản ghi trên mỗi trang
#
#         # Tính toán bản ghi bắt đầu và kết thúc
#         start = (page - 1) * per_page
#         end = start + per_page
#
#         # Lấy dữ liệu từ database
#         with sqlite3.connect("parking.db") as conn:
#             conn.row_factory = sqlite3.Row
#             cursor = conn.cursor()
#             cursor.execute('SELECT * FROM PriceConvention')
#             rows = cursor.fetchall()
#             total_items = len(rows)
#             total_pages = (total_items + per_page - 1) // per_page
#             data = [dict(row) for row in rows[start:end]]
#
#         # Truyền dữ liệu vào giao diện
#         return render_template(
#             'price_convention.html',
#             price_conventions=data,
#             current_page=page,
#             total_pages=total_pages
#         )
#     except Exception as e:
#         return render_template('price_convention.html', error=str(e))

@app.route('/price_convention', methods=['GET'])
def get_all_price_conventions():
    try:
        # Lấy trang hiện tại từ query string
        page = request.args.get('page', 1, type=int)
        per_page = 5  # Số lượng bản ghi trên mỗi trang

        # Tính toán bản ghi bắt đầu và kết thúc
        start = (page - 1) * per_page
        end = start + per_page

        # Lấy dữ liệu từ database
        with sqlite3.connect("parking.db") as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT PriceConvention.*, VehicleType.Name AS VehicleTypeName
                FROM PriceConvention
                JOIN VehicleType ON PriceConvention.VehicleTypeID = VehicleType.ID
            ''')
            rows = cursor.fetchall()
            total_items = len(rows)
            total_pages = (total_items + per_page - 1) // per_page
            data = [dict(row) for row in rows[start:end]]

        # Truyền dữ liệu vào giao diện
        return render_template(
            'price_convention.html',
            price_conventions=data,
            current_page=page,
            total_pages=total_pages
        )
    except Exception as e:
        return render_template('price_convention.html', error=str(e))


## API CHO VEHICALE_TYPE
#API POSTMAN GET VEHICLE_TYPE
# @app.route('/vehicle_type', methods=['GET'])
# def get_all_vehicle_types():
#     try:
#         with sqlite3.connect("parking.db") as conn:
#             conn.row_factory = sqlite3.Row
#             cursor = conn.cursor()
#             cursor.execute('SELECT * FROM VehicleType')
#             rows = cursor.fetchall()
#             data = [dict(row) for row in rows]
#         return jsonify(data), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

#API GET VEHICLE_TYPE HTML
@app.route('/vehicle_type', methods=['GET'])
def get_all_vehicle_type():
    try:
        # Lấy trang hiện tại từ query string
        page = request.args.get('page', 1, type=int)
        per_page = 5  # Số lượng bản ghi trên mỗi trang

        # Tính toán bản ghi bắt đầu và kết thúc
        start = (page - 1) * per_page
        end = start + per_page

        # Lấy dữ liệu từ database
        with sqlite3.connect("parking.db") as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM VehicleType')
            rows = cursor.fetchall()
            total_items = len(rows)
            total_pages = (total_items + per_page - 1) // per_page
            data = [dict(row) for row in rows[start:end]]

        # Truyền dữ liệu vào giao diện
        return render_template(
            'vehicle_type.html',
            vehicle_type=data,
            current_page=page,
            total_pages=total_pages
        )
    except Exception as e:
        return render_template('vehicle_type.html', error=str(e))


#API POSTMAN POST VEHICLE_TYPE
# @app.route('/vehicle_type', methods=['POST'])
# def add_vehicle_type():
#     data = request.get_json()
#     name = data.get('Name')
#     with sqlite3.connect("parking.db") as conn:
#         cursor = conn.cursor()
#         cursor.execute("INSERT INTO VehicleType (Name) VALUES (?)", (name,))
#         conn.commit()
#         return jsonify({"message": "Vehicle type added successfully."}), 201

#API Post VEHICLE_TYPE html
@app.route('/vehicle_type', methods=['POST'])
def add_vehicle_type():
    try:
        # Lấy dữ liệu từ form
        name = request.form.get('Name')

        # Thêm vào database
        with sqlite3.connect("parking.db") as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO VehicleType (Name)
            VALUES (?)
            ''', (name,))
            conn.commit()
        return redirect('/vehicle_type')  # Chuyển hướng về trang chính
    except Exception as e:
        return render_template('vehicle_type.html', error=str(e))



#API POSTMAN UPDATE VEHICLE_TYPE
# @app.route('/vehicle_type/<int:id>', methods=['PUT'])
# def update_vehicle_type(id):
#     try:
#         data = request.json
#         with sqlite3.connect("parking.db") as conn:
#             cursor = conn.cursor()
#             cursor.execute('''
#                 UPDATE VehicleType
#                 SET Name = ?
#                 WHERE ID = ?
#             ''', (data['Name'], id))
#             conn.commit()
#         return jsonify({"message": "VehicleType updated successfully!"}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

#API UPDATE VEHICLE_TYPE TRÊN HTML
@app.route('/vehicle_type/update/<int:id>', methods=['POST'])
def update_vehicle_type(id):
    try:
        # Lấy dữ liệu từ form
        name = request.form.get('Name')

        # Cập nhật dữ liệu trong database
        with sqlite3.connect("parking.db") as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE VehicleType
                SET Name = ?
                WHERE ID = ?
            ''', (name, id))
            conn.commit()

        return redirect('/vehicle_type')  # Chuyển hướng về trang chính
    except Exception as e:
        return render_template('vehicle_type.html', error=str(e))

#API DELETE VEHICLE_TYPE HTML
@app.route('/vehicle_type/delete/<int:id>', methods=['POST'])
def delete_vehicle_type(id):
    try:
        # Xóa dữ liệu trong database
        with sqlite3.connect("parking.db") as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM VehicleType WHERE ID = ?', (id,))
            conn.commit()

        # Sau khi xóa, lấy lại dữ liệu để render giao diện
        return redirect('/vehicle_type')  # Chuyển hướng lại trang Price Convention
    except Exception as e:
        # Lấy lại dữ liệu để tránh lỗi giao diện
        with sqlite3.connect("parking.db") as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM VehicleType')
            rows = cursor.fetchall()
            data = [dict(row) for row in rows]
            total_pages = 1  # Hoặc tính toán lại số trang nếu cần
        return render_template(
            'vehicle_type.html',
            vehicle_type=data,
            current_page=1,
            total_pages=total_pages,
            error=str(e)
        )

#API POSTMAN DELETE VEHICLE_TYPE
# @app.route('/vehicle_type/<int:id>', methods=['DELETE'])
# def delete_vehicle_type(id):
#     try:
#         with sqlite3.connect("parking.db") as conn:
#             cursor = conn.cursor()
#             cursor.execute('DELETE FROM VehicleType WHERE ID = ?', (id,))
#             conn.commit()
#         return jsonify({"message": "VehicleType deleted successfully!"}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400


##API CHO ACTUAL PARKING FEE

# API POSTMAN CHO POST actual_parking_fee
# @app.route('/actual_parking_fee', methods=['POST'])
# def add_parking_fee():
#     data = request.get_json()
#     vehicle_type_id = data.get('VehicleTypeID')
#     plate_number = data.get('PlateNumber')
#     entry_time = data.get('EntryTime')
#     exit_time = data.get('ExitTime')
#     status = data.get('Status')
#     total_fee = data.get('TotalFee')
#     note = data.get('Note')
#     with sqlite3.connect("park ing.db") as conn:
#         cursor = conn.cursor()
#         cursor.execute('''
#         INSERT INTO ActualParkingFee (VehicleTypeID, PlateNumber, EntryTime, ExitTime, Status, TotalFee, Note)
#         VALUES (?, ?, ?, ?, ?, ?, ?)
#         ''', (vehicle_type_id, plate_number, entry_time, exit_time, status, total_fee, note))
#         conn.commit()
#         return jsonify({"message": "Actual parking fee added successfully."}), 201

@app.route('/actual_parking_fee', methods=['POST'])
def add_actual_parking_fee():
    try:
        # Lấy dữ liệu từ form
        vehicle_id = request.form.get('VehicleTypeID')
        plate_number = request.form.get('PlateNumber')
        entry_time = request.form.get('EntryTime')
        exit_time = request.form.get('ExitTime')
        status = request.form.get('Status')
        total_fee = request.form.get('TotalFee')
        note = request.form.get('Note')

        # Thêm vào database
        with sqlite3.connect("parking.db") as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO ActualParkingFee (VehicleTypeID, PlateNumber, EntryTime, ExitTime, Status, TotalFee, Note)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (vehicle_id, plate_number, entry_time, exit_time, status, total_fee, note))
            conn.commit()
        return redirect('/actual_parking_fee_processing')  # Chuyển hướng về trang chính
    except Exception as e:
        return render_template('actual_parking_fee_processing.html', error=str(e))

#API POSTMAN cho get Actual_parking_fee
# @app.route('/actual_parking_fee', methods=['GET'])
# def get_all_actual_parking_fees():
#     try:
#         with sqlite3.connect("parking.db") as conn:
#             conn.row_factory = sqlite3.Row
#             cursor = conn.cursor()
#             cursor.execute('SELECT * FROM ActualParkingFee')
#             rows = cursor.fetchall()
#             data = [dict(row) for row in rows]
#         return jsonify(data), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

@app.route('/actual_parking_fee', methods=['GET'])
def get_all_actual_parking_fee():
    try:
        # Lấy trang hiện tại từ query string
        page = request.args.get('page', 1, type=int)
        per_page = 10  # Số lượng bản ghi trên mỗi trang

        # Tính toán bản ghi bắt đầu và kết thúc
        start = (page - 1) * per_page
        end = start + per_page

        # Lấy dữ liệu từ database
        with sqlite3.connect("parking.db") as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT ActualParkingFee.*, VehicleType.Name AS VehicleTypeName
                FROM ActualParkingFee
                JOIN VehicleType ON ActualParkingFee.VehicleTypeID = VehicleType.ID
            ''')
            rows = cursor.fetchall()
            total_items = len(rows)
            total_pages = (total_items + per_page - 1) // per_page
            data = [dict(row) for row in rows[start:end]]

        # Truyền dữ liệu vào giao diện
        return render_template(
            'actual_parking_fee.html',
            actual_parking_fee=data,
            current_page=page,
            total_pages=total_pages
        )
    except Exception as e:
        return render_template('actual_parking_fee.html', error=str(e))



@app.route('/actual_parking_fee_processing', methods=['GET'])
def get_all_actual_parking_fee_processing():
    try:
        # Lấy trang hiện tại từ query string
        page = request.args.get('page', 1, type=int)
        per_page = 10  # Số lượng bản ghi trên mỗi trang

        # Tính toán bản ghi bắt đầu và kết thúc
        start = (page - 1) * per_page
        end = start + per_page

        # Lấy dữ liệu từ database
        with sqlite3.connect("parking.db") as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT ActualParkingFee.*, VehicleType.Name AS VehicleTypeName
                FROM ActualParkingFee
                JOIN VehicleType ON ActualParkingFee.VehicleTypeID = VehicleType.ID
                WHERE ActualParkingFee.Status = 'Đang gửi xe'
            ''')
            rows = cursor.fetchall()
            total_items = len(rows)
            total_pages = (total_items + per_page - 1) // per_page
            data = [dict(row) for row in rows[start:end]]

        # Truyền dữ liệu vào giao diện
        return render_template(
            'actual_parking_fee_processing.html',
            actual_parking_fee=data,
            current_page=page,
            total_pages=total_pages
        )
    except Exception as e:
        return render_template('actual_parking_fee_processing.html', error=str(e))
# API POSTMAN DELETE actual_parking_fee
# @app.route('/actual_parking_fee/ApiDelete/<int:id>', methods=['DELETE'])
# def APIdelete_actual_parking_fee(id):
#     try:
#         with sqlite3.connect("parking.db") as conn:
#             cursor = conn.cursor()
#             cursor.execute('DELETE FROM ActualParkingFee WHERE ID = ?', (id,))
#             conn.commit()
#         return jsonify({"message": "ActualParkingFee deleted successfully!"}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400
#API DELETE CHO HTML
@app.route('/actual_parking_fee/delete/<int:id>', methods=['POST'])
def delete_actual_parking_fee(id):
    try:
        # Xóa dữ liệu trong database
        with sqlite3.connect("parking.db") as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM ActualParkingFee WHERE ID = ?', (id,))
            conn.commit()

        # Sau khi xóa, lấy lại dữ liệu để render giao diện
        return redirect('/actual_parking_fee')  # Chuyển hướng lại trang
    except Exception as e:
        # Lấy lại dữ liệu để tránh lỗi giao diện
        with sqlite3.connect("parking.db") as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM ActualParkingFee')
            rows = cursor.fetchall()
            data = [dict(row) for row in rows]
            total_pages = 1  # Hoặc tính toán lại số trang nếu cần
        return render_template(
            'actual_parking_fee.html',
            actual_parking_fee=data,
            current_page=1,
            total_pages=total_pages,
            error=str(e)
        )

@app.route('/account', methods=['POST'])
def add_account():
    data = request.get_json()
    username = data.get('Username')
    password = data.get('Password')
    with sqlite3.connect("parking.db") as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Account (Username, Password) VALUES (?, ?)", (username, password))
        conn.commit()
        return jsonify({"message": "Account created successfully."}), 201



@app.route('/')
def home():
    return render_template('base.html')

@app.after_request
def disable_caching(response):
    response.headers['Cache-Control'] = 'no-store'
    return response

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
