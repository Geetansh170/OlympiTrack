from dbcrud import create_entry, delete_entry, read_entries, update_entry

def display_and_modify_table(st, table_name):
    st.write(f"### Table: {table_name}")
    df = read_entries(table_name)
    st.dataframe(df)

    tab1, tab2, tab3, tab4 = st.tabs(["Lookup", "Add Entry", "Update Entry", "Delete Entry"])
    
    with tab1:
        st.write("### Lookup Entries")
        with st.form(f"lookup_form_{table_name}"):
            search_values = {}
            st.write("Enter search values for columns (leave blank to ignore a column):")
            for column in df.columns:
                search_values[column] = st.text_input(f"Search value for {column}", key=f"search_{column}_{table_name}")
            submitted = st.form_submit_button("Search")
            if submitted:
                filtered_df = df.copy()
                for column, value in search_values.items():
                    if value:  # Apply filter only if a value is provided
                        filtered_df = filtered_df[filtered_df[column].astype(str).str.lower().str.contains(value.lower(), na=False)]
                if not filtered_df.empty:
                    st.dataframe(filtered_df)
                else:
                    st.warning("No matching entries found.")

    # Add Entry Tab
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

    # Update Entry Tab
    with tab3:
        st.write("### Update Existing Entry")
        with st.form(f"update_entry_form_{table_name}"):
            condition = st.text_input("Update Condition (e.g., id = 1 AND status = 'active')", key=f"update_condition_{table_name}")
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
            condition = st.text_input("Delete Condition (e.g., sport = 'athletics' and edition_id = 5)", key=f"delete_condition_{table_name}")
            submitted = st.form_submit_button("Delete Entry")
            if submitted:
                delete_entry(table_name, condition)
                st.success("Entry deleted successfully!")
