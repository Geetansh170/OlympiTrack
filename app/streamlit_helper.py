from dbcrud import create_entry, delete_entry, read_entries, update_entry

def display_and_modify_table(st, table_name):
    st.write(f"### Table: {table_name}")
    df = read_entries(table_name)
    st.dataframe(df)

    tab1, tab2, tab3, tab4 = st.tabs(["Lookup", "Add Entry", "Update Entry", "Delete Entry"])
    
    with tab1:
        st.write("### Lookup Entries")
        with st.form(f"lookup_form_{table_name}"):
            search_column = st.selectbox("Search by Column", options=df.columns, key=f"search_column_{table_name}")
            search_value = st.text_input(f"Enter value for {search_column}", key=f"search_value_{table_name}")
            submitted = st.form_submit_button("Search")
            if submitted:
                filtered_df = df[df[search_column].astype(str).str.lower().str.contains(search_value.lower(), na=False)]
                if not filtered_df.empty:
                    st.dataframe(filtered_df)
                else:
                    st.warning("No matching entries found.")

    with tab2:
        st.write("### Add New Entry")
        with st.form(f"add_entry_form_{table_name}"):
            new_entry = {}
            for column in df.columns:
                new_entry[column] = st.text_input(f"{column}", key=f"add_{column}_{table_name}")
            submitted = st.form_submit_button("Add Entry")
            if submitted:
                create_entry(table_name, new_entry)
                st.success("Entry added successfully!")

    with tab3:
        st.write("### Update Existing Entry")
        with st.form(f"update_entry_form_{table_name}"):
            condition = st.text_input("Update Condition (e.g., id = 1)", key=f"update_condition_{table_name}")
            update_data = {}
            for column in df.columns:
                update_data[column] = st.text_input(f"New Value for {column}", key=f"update_{column}_{table_name}")
            submitted = st.form_submit_button("Update Entry")
            if submitted:
                update_entry(table_name, update_data, condition)
                st.success("Entry updated successfully!")

    with tab4:
        st.write("### Delete Entry")
        with st.form(f"delete_entry_form_{table_name}"):
            condition = st.text_input(f"Delete Condition for {table_name} (e.g., id = 1)", key=f"delete_condition_{table_name}")
            submitted = st.form_submit_button("Delete Entry")
            if submitted:
                delete_entry(table_name, condition)
                st.success("Entry deleted successfully!")
