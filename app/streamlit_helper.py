from dbcrud import create_entry, delete_entry, read_entries, update_entry


def display_and_modify_table(st, table_name):
    st.write(f"### Table: {table_name}")
    df = read_entries(table_name)
    st.dataframe(df)

    with st.expander("Add New Entry"):
        with st.form(f"add_entry_{table_name}"):
            new_entry = {}
            for column in df.columns:
                new_entry[column] = st.text_input(f"{column}", "")
            submitted = st.form_submit_button("Add Entry")
            if submitted:
                create_entry(table_name, new_entry)
                st.success("Entry added successfully!")

    with st.expander("Update Entry"):
        with st.form(f"update_entry_{table_name}"):
            condition = st.text_input("Update Condition (e.g., id = 1)")
            update_data = {}
            for column in df.columns:
                update_data[column] = st.text_input(f"New Value for {column}", "")
            submitted = st.form_submit_button("Update Entry")
            if submitted:
                update_entry(table_name, update_data, condition)
                st.success("Entry updated successfully!")

    with st.expander("Delete Entry"):
        condition = st.text_input(f"Delete Condition for {table_name} (e.g., id = 1)")
        if st.button(f"Delete Entry from {table_name}"):
            delete_entry(table_name, condition)
            st.success("Entry deleted successfully!")